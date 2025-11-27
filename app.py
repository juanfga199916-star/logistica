import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import openrouteservice
import hashlib
import json
from math import radians, cos, sin, asin, sqrt
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import time
import random

# --- CONFIG ---
st.set_page_config(page_title="Panel de Control de Rutas", page_icon="🗺️", layout="wide")

# --- SESSION STATE INIT ---
if 'puntos' not in st.session_state:
    st.session_state.puntos = []
if 'map_center' not in st.session_state:
    st.session_state.map_center = [4.60971, -74.08175]
if 'centro' not in st.session_state:
    st.session_state.centro = None
if 'seleccionando_centro' not in st.session_state:
    st.session_state.seleccionando_centro = False
if 'route_geojson' not in st.session_state:
    st.session_state.route_geojson = None
if 'route_metrics' not in st.session_state:
    st.session_state.route_metrics = None
if 'fleet' not in st.session_state:
    st.session_state.fleet = []
if 'last_fingerprint' not in st.session_state:
    st.session_state.last_fingerprint = None

# --- CONFIGURAR ORS ---
# Si tienes ORS key ponla aquí; si no, deja vacía "" para usar fallback.
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImY5MTA5MmE2NzVmZDRhYjBhMTk4YjZiNWNiMWY2YjQzIiwiaCI6Im11cm11cjY0In0="  # <-- pega tu ORS key si deseas geometrías reales
try:
    client = openrouteservice.Client(key=ORS_API_KEY) if ORS_API_KEY else None
except Exception:
    client = None

# --- HELPERS ---
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 6371 * 2 * asin(sqrt(a))

def geocode_address(address, city_hint=None, retries=2):
    geolocator = Nominatim(user_agent="routing_app")
    q = address if pd.isna(city_hint) else f"{address}, {city_hint}"
    for _ in range(retries):
        try:
            loc = geolocator.geocode(q, timeout=10)
            if loc:
                return loc.latitude, loc.longitude
        except Exception:
            time.sleep(1)
    return None, None

def crear_tabla_de_pedidos(puntos):
    if not puntos:
        cols = ['ID Pedido','Destino','Prioridad','Peso (kg)','Volumen (m³)','Service (min)','TW Inicio','TW Fin','Latitud','Longitud']
        return pd.DataFrame(columns=cols)
    data = {
        'ID Pedido': [p.get('nombre', f"Punto {i+1}") for i,p in enumerate(puntos)],
        'Destino': [p.get('direccion', '') for p in puntos],
        'Prioridad': [p.get('prioridad','Media') for p in puntos],
        'Peso (kg)': [p.get('peso',0.0) for p in puntos],
        'Volumen (m³)': [p.get('volumen',0.0) for p in puntos],
        'Service (min)': [p.get('service_time',5) for p in puntos],
        'TW Inicio': [p.get('tw_start','00:00') for p in puntos],
        'TW Fin': [p.get('tw_end','23:59') for p in puntos],
        'Latitud': [p.get('lat','') for p in puntos],
        'Longitud': [p.get('lon','') for p in puntos]
    }
    return pd.DataFrame(data)

def fingerprint(centro, puntos, fleet):
    obj = {'centro': centro, 'puntos': [(p.get('lat'), p.get('lon'), p.get('peso'), p.get('volumen'), p.get('tw_start'), p.get('tw_end')) for p in puntos], 'fleet': fleet}
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def parse_single_sheet_excel(df_raw):
    """
    Detecta secciones en un DataFrame leido sin encabezados claros:
    - CEDI arriba
    - 'CARACTERISTICAS DE FLOTA' línea separadora
    - 'CARACTERISTICAS DE PEDIDOS' línea separadora
    Devuelve: cedi_info (dict), fleet_list (list of dicts), pedidos_df (DataFrame)
    """
    # Normalizar: convertir todo a strings para buscar títulos
    df0 = df_raw.fillna('').astype(str)
    nrows, ncols = df0.shape

    fleet_start = None
    pedidos_start = None
    cedi = {}
    # Buscar filas con las etiquetas
    for i in range(min(10, nrows)):
        rowstr = " ".join(df0.iloc[i].str.strip().str.lower().tolist())
        if 'cedi' in rowstr or 'cedi de distribucion' in rowstr:
            # tomar la siguiente fila no vacía como info
            for j in range(i+1, min(i+5, nrows)):
                # verificar si la fila tiene algún valor
                if any([str(x).strip() != '' for x in df_raw.iloc[j].tolist()]):
                    first = df_raw.iloc[j,0] if pd.notna(df_raw.iloc[j,0]) else ''
                    second = df_raw.iloc[j,1] if df_raw.shape[1]>1 and pd.notna(df_raw.iloc[j,1]) else ''
                    cedi = {'name': str(first).strip(), 'direccion': str(second).strip()}
                    break
        if 'caracteristicas de flota' in rowstr or 'características de flota' in rowstr:
            fleet_start = i+1
        if 'caracteristicas de pedidos' in rowstr or 'características de pedidos' in rowstr:
            pedidos_start = i+1

    # Parse fleet table: asumimos filas hasta línea vacía
    fleet = []
    if fleet_start:
        for i in range(fleet_start, nrows):
            row_vals = df_raw.iloc[i].tolist()
            if all([ (str(x).strip()=='' ) for x in row_vals ]):
                break
            vals = [ (str(x).strip()) for x in row_vals if str(x).strip()!='' ]
            if len(vals) >= 4:
                tipo = vals[0]
                try:
                    kg = float(vals[1].replace(',','.'))
                except:
                    try:
                        kg = float(vals[1])
                    except:
                        kg = 0.0
                try:
                    m3 = float(vals[2].replace(',','.'))
                except:
                    m3 = 0.0
                try:
                    speed = float(vals[3].replace(',','.'))
                except:
                    speed = 40.0
                shift_start = vals[4] if len(vals) >= 5 else '07:00'
                shift_end = vals[5] if len(vals) >= 6 else '19:00'
                fleet.append({'tipo': tipo, 'capacity_kg': kg, 'capacity_m3': m3, 'speed_kmh': speed, 'shift_start': shift_start, 'shift_end': shift_end})
    # Parse pedidos table: encontrar header y leer with pandas
    pedidos_df = None
    if pedidos_start:
        # from pedidos_start find header row with 'nombre' or 'nombre_pedido'
        header_row = None
        for i in range(pedidos_start, min(pedidos_start+8, nrows)):
            row_lower = [str(x).strip().lower() for x in df0.iloc[i].tolist()]
            if any('nombre' in c for c in row_lower) and any('peso' in c for c in row_lower):
                header_row = i
                break
        if header_row is not None:
            pedidos_df = df_raw.iloc[header_row: , :].copy()
            pedidos_df.columns = [str(c).strip().lower() for c in pedidos_df.iloc[0].tolist()]
            pedidos_df = pedidos_df.iloc[1:].reset_index(drop=True)
            pedidos_df = pedidos_df.dropna(how='all')
    return cedi, fleet, pedidos_df

# --- VRPTW solver (OR-Tools) simplified ---
def solve_vrptw(centro, puntos, fleet_list, time_limit_seconds=20):
    """
    Centro: [lat, lon]
    puntos: list of dicts each with lat, lon, peso, volumen, tw_start, tw_end, service_time
    fleet_list: list of dicts each with capacity_kg, capacity_m3, speed_kmh, shift_start, shift_end, tipo
    Returns: routes_info: list of dicts {'vehicle_idx', 'vehicle_tipo', 'sequence' (node indices), 'coords'}, metrics_est
    """
    if not centro or not puntos or not fleet_list:
        return None, None

    # Nodes: depot(0) + pedidos 1..N
    nodes = [{'lat': centro[0], 'lon': centro[1], 'demand_w': 0, 'demand_v': 0, 'service': 0, 'tw_start':'00:00', 'tw_end':'23:59'}]
    for p in puntos:
        nodes.append({
            'lat': p['lat'], 'lon': p['lon'],
            'demand_w': p.get('peso',0.0),
            'demand_v': p.get('volumen',0.0),
            'service': p.get('service_time',5),
            'tw_start': p.get('tw_start','08:00'),
            'tw_end': p.get('tw_end','18:00')
        })
    N = len(nodes)

    # Build time and distance matrices using average speed of fleet
    avg_speed = np.mean([f.get('speed_kmh',40) for f in fleet_list])
    time_matrix = [[0]*N for _ in range(N)]
    dist_matrix = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i==j:
                time_matrix[i][j] = 0
                dist_matrix[i][j] = 0.0
            else:
                km = haversine_km(nodes[i]['lat'], nodes[i]['lon'], nodes[j]['lat'], nodes[j]['lon'])
                dist_matrix[i][j] = km
                travel_min = (km / max(avg_speed,0.1))*60.0
                time_matrix[i][j] = int(round(travel_min))

    # OR-Tools data
    demands_w = [int(round(n['demand_w'])) for n in nodes]  # kg
    demands_v = [int(round(n['demand_v']*1000)) for n in nodes]  # m3 -> liters scale
    service_times = [int(round(n['service'])) for n in nodes]
    time_windows = [(int(0), int(24*60)) for n in nodes]
    for idx, n in enumerate(nodes):
        try:
            ts = time_str_to_minutes(n['tw_start']) if 'tw_start' in n else 0
            te = time_str_to_minutes(n['tw_end']) if 'tw_end' in n else 24*60
        except:
            ts, te = 0, 24*60
        time_windows[idx] = (ts, te)

    num_vehicles = len(fleet_list)
    vehicle_cap_w = [int(round(f['capacity_kg'])) for f in fleet_list]
    vehicle_cap_v = [int(round(f['capacity_m3']*1000)) for f in fleet_list]  # scale
    depot_index = 0

    manager = pywrapcp.RoutingIndexManager(N, num_vehicles, depot_index)
    routing = pywrapcp.RoutingModel(manager)

    # Transit callback (time)
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return time_matrix[from_node][to_node] + service_times[to_node]
    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add time dimension
    time_dim_name = 'Time'
    routing.AddDimension(
        transit_callback_index,
        24*60,  # allow waiting
        24*60,
        False,
        time_dim_name
    )
    time_dimension = routing.GetDimensionOrDie(time_dim_name)

    # Set time windows for each location
    for node_idx in range(N):
        index = manager.NodeToIndex(node_idx)
        start, end = time_windows[node_idx]
        time_dimension.CumulVar(index).SetRange(start, end)

    # Set vehicle start time windows from fleet shifts
    for v in range(num_vehicles):
        start_idx = routing.Start(v)
        shift_s = time_str_to_minutes(fleet_list[v].get('shift_start','07:00'))
        shift_e = time_str_to_minutes(fleet_list[v].get('shift_end','19:00'))
        time_dimension.CumulVar(start_idx).SetRange(shift_s, shift_e)

    # Capacity weight callback
    def demand_w_callback(from_index):
        node = manager.IndexToNode(from_index)
        return demands_w[node]
    demand_w_idx = routing.RegisterUnaryTransitCallback(demand_w_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_w_idx,
        0,
        vehicle_cap_w,
        True,
        'CapacityW'
    )

    def demand_v_callback(from_index):
        node = manager.IndexToNode(from_index)
        return demands_v[manager.IndexToNode(from_index)]
    demand_v_idx = routing.RegisterUnaryTransitCallback(demand_v_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_v_idx,
        0,
        vehicle_cap_v,
        True,
        'CapacityV'
    )

    # Search params
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = time_limit_seconds
    search_parameters.log_search = False

    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        return None, None

    # Extract routes
    routes_info = []
    total_km = 0.0
    total_min = 0.0
    for v in range(num_vehicles):
        idx = routing.Start(v)
        route_nodes = []
        route_km = 0.0
        route_min = 0.0
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            route_nodes.append(node)
            next_idx = solution.Value(routing.NextVar(idx))
            if routing.IsEnd(next_idx):
                break
            next_node = manager.IndexToNode(next_idx)
            route_km += dist_matrix[node][next_node]
            route_min += time_matrix[node][next_node] + service_times[next_node]
            idx = next_idx
        # build coords sequence for the route (include depot start and end)
        if len(route_nodes) <= 1:
            routes_info.append({'vehicle_idx': v, 'vehicle_tipo': fleet_list[v].get('tipo','vehiculo'), 'sequence': [], 'coords': []})
            continue
        seq = []
        seq.append([nodes[0]['lon'], nodes[0]['lat']])
        for nidx in route_nodes:
            if nidx==0:
                continue
            seq.append([nodes[nidx]['lon'], nodes[nidx]['lat']])
        seq.append([nodes[0]['lon'], nodes[0]['lat']])
        routes_info.append({'vehicle_idx': v, 'vehicle_tipo': fleet_list[v].get('tipo','vehiculo'), 'sequence': route_nodes, 'coords': seq})
        total_km += route_km
        total_min += route_min

    metrics_est = {'distance_km': total_km, 'time_min': total_min}
    return routes_info, metrics_est

def time_str_to_minutes(tstr):
    try:
        h,m = map(int, str(tstr).split(':'))
        return h*60 + m
    except:
        # try without leading zero '7:00'
        try:
            parts = str(tstr).split(':')
            if len(parts)==2:
                return int(parts[0])*60 + int(parts[1])
        except:
            return 0
    return 0

def request_ors_route(coords_sequence):
    if client is None or not coords_sequence or len(coords_sequence) < 2:
        return None, None
    try:
        route = client.directions(coords_sequence, profile='driving-car', format='geojson')
        feat = route.get('features', [None])[0]
        if not feat:
            return None, None
        segments = feat.get('properties', {}).get('segments', [])
        total_distance = sum(seg.get('distance', 0) for seg in segments)
        total_duration = sum(seg.get('duration', 0) for seg in segments)
        return feat, {"distance_m": total_distance, "duration_s": total_duration}
    except Exception as e:
        st.warning(f"ORS request failed: {e}")
        return None, None

# --- UI ---
st.title("🗺️ Panel de Control para Optimización de Rutas (Formato único Excel)")
st.write("Sube el archivo Excel (misma hoja con secciones: CEDI, CARACTERISTICAS DE FLOTA, CARACTERISTICAS DE PEDIDOS).")

# Sidebar: cost input
with st.sidebar:
    st.header("Parámetros")
    cost_per_km = st.number_input("Costo por kilómetro (unidad monetaria)", min_value=0.0, value=0.5, step=0.1)
    st.write("Si quieres usar ORS activa tu API key en el código (si la tienes).")

with st.sidebar:
    st.header("📂 Cargar archivo (única hoja con secciones)")
    uploaded = st.file_uploader("Selecciona .xlsx", type=['xlsx'])
    if uploaded:
        try:
            # Leer hoja completa sin forzar encabezado
            df_raw = pd.read_excel(uploaded, header=None, dtype=str)
            cedi_info, fleet_list, pedidos_df = parse_single_sheet_excel(df_raw)
            if cedi_info:
                st.markdown(f"**CEDI:** {cedi_info.get('name','')} — {cedi_info.get('direccion','')}")
                # set map center to cedi if geocodable
                if cedi_info.get('direccion'):
                    latc, lonc = geocode_address(cedi_info.get('direccion'), city_hint=None)
                    if latc:
                        st.session_state.map_center = [latc, lonc]
                        # If centro not set, set it
                        if not st.session_state.centro:
                            st.session_state.centro = [latc, lonc]
            if fleet_list:
                st.session_state.fleet = fleet_list
                st.markdown("**Fleet loaded:**")
                st.dataframe(pd.DataFrame(fleet_list))
            else:
                st.info("No se detectó tabla de flota. Añádela en el Excel según ejemplo.")
            if pedidos_df is not None:
                st.markdown("**Pedidos (raw)**")
                st.dataframe(pedidos_df.head(30))
                # normalize column names
                pedidos_df.columns = [c.strip().lower() for c in pedidos_df.columns]
                # geocode or use lat/lon
                loaded = 0
                new_points = []
                for i, row in pedidos_df.iterrows():
                    # read fields safely
                    nombre = row.get('nombre_pedido') or row.get('nombre') or f'Pedido {i+1}'
                    peso = 0.0
                    volumen = 0.0
                    try:
                        if pd.notna(row.get('peso', None)):
                            peso = float(str(row.get('peso')).replace(',','.'))
                    except:
                        peso = 0.0
                    try:
                        if pd.notna(row.get('volumen', None)):
                            volumen = float(str(row.get('volumen')).replace(',','.'))
                    except:
                        volumen = 0.0
                    prioridad = row.get('prioridad','Media')
                    # normalize time fields
                    tws = row.get('tw_start') or row.get('tw_start'.lower()) if False else row.get('tw_start','08:00')
                    twf = row.get('tw_end') or row.get('tw_end'.lower()) if False else row.get('tw_end','18:00')
                    # fallbacks for different column names
                    if not tws and 'tw_start' in pedidos_df.columns:
                        tws = pedidos_df.at[i,'tw_start']
                    if not twf and 'tw_end' in pedidos_df.columns:
                        twf = pedidos_df.at[i,'tw_end']
                    ciudad = row.get('ciudad', None)
                    direccion = row.get('direccion', None) or row.get('dirección', None)
                    lat, lon = None, None
                    # If lat/lon exist as columns
                    if 'lat' in pedidos_df.columns and ('lon' in pedidos_df.columns or 'longitud' in pedidos_df.columns):
                        lon_col = 'lon' if 'lon' in pedidos_df.columns else 'longitud'
                        try:
                            lat = float(row.get('lat'))
                            lon = float(row.get(lon_col))
                        except:
                            lat, lon = None, None
                    # else geocode from direccion + ciudad
                    if (lat is None or lon is None) and direccion:
                        lat, lon = geocode_address(str(direccion), city_hint=ciudad)
                        time.sleep(1)  # polite delay
                    if lat and lon:
                        new_points.append({
                            'nombre': nombre,
                            'peso': peso,
                            'volumen': volumen,
                            'prioridad': prioridad,
                            'tw_start': str(tws),
                            'tw_end': str(twf),
                            'direccion': direccion,
                            'lat': float(lat),
                            'lon': float(lon),
                            'service_time': int(float(row.get('service_time',5))) if 'service_time' in pedidos_df.columns else 5
                        })
                        loaded += 1
                st.success(f"{loaded} pedidos con coordenadas cargados desde Excel.")
                st.session_state.puntos = new_points
            else:
                st.info("No se detectó tabla de pedidos. Revisa el formato Excel.")
        except Exception as e:
            st.error(f"Error leyendo/parsing archivo: {e}")

    st.divider()
    st.subheader("Centro")
    if st.button("📍 Seleccionar Centro en mapa"):
        st.session_state.seleccionando_centro = True
        st.info("Haz clic en el mapa para definir el centro.")

    st.divider()
    st.subheader("Control de cálculo")
    if st.button("🚀 Calcular rutas (VRPTW)"):
        if not st.session_state.centro:
            st.warning("Define centro de distribución primero (o déjalo en CEDI).")
        elif not st.session_state.puntos:
            st.warning("No hay pedidos cargados.")
        elif not st.session_state.fleet:
            st.warning("No hay flota definida. Añade la tabla de flota en el Excel.")
        else:
            # fingerprint
            fp = fingerprint(st.session_state.centro, st.session_state.puntos, st.session_state.fleet)
            if fp == st.session_state.last_fingerprint and st.session_state.route_geojson:
                st.info("La ruta ya está calculada y actualizada.")
            else:
                with st.spinner("Resolviendo VRPTW (OR-Tools)..."):
                    routes_info, metrics_est = solve_vrptw(st.session_state.centro, st.session_state.puntos, st.session_state.fleet, time_limit_seconds=20)
                    if not routes_info:
                        st.error("No se encontró solución factible con las restricciones dadas.")
                    else:
                        # solicitar ORS para cada ruta si client existe, sino fallback lineas rectas
                        features = []
                        per_route_metrics = []  # each: {'vehicle_idx','vehicle_tipo','distance_km','duration_s'}
                        total_distance_m = 0
                        total_duration_s = 0
                        for r in routes_info:
                            coords = r.get('coords', [])
                            if not coords:
                                continue
                            # coords are [lon, lat] pairs
                            if client:
                                feat, m = request_ors_route(coords)
                                if feat and m:
                                    features.append(feat)
                                    route_distance_m = m['distance_m']
                                    route_duration_s = m['duration_s']
                                else:
                                    # fallback: estimate from straight-line segments
                                    route_distance_km = 0.0
                                    for k in range(len(coords)-1):
                                        lon1, lat1 = coords[k]
                                        lon2, lat2 = coords[k+1]
                                        route_distance_km += haversine_km(lat1, lon1, lat2, lon2)
                                    route_distance_m = route_distance_km * 1000
                                    # estimate duration using average fleet speed (m/s)
                                    avg_speed_kmh = np.mean([f.get('speed_kmh',40) for f in st.session_state.fleet]) if st.session_state.fleet else 40
                                    route_duration_s = (route_distance_km / max(avg_speed_kmh,0.1)) * 3600
                                    # add GeoJSON line
                                    ls = {"type":"Feature","properties":{"vehicle":r['vehicle_tipo']},"geometry":{"type":"LineString","coordinates":coords}}
                                    features.append(ls)
                            else:
                                # No ORS: compute straight-line distance
                                route_distance_km = 0.0
                                for k in range(len(coords)-1):
                                    lon1, lat1 = coords[k]
                                    lon2, lat2 = coords[k+1]
                                    route_distance_km += haversine_km(lat1, lon1, lat2, lon2)
                                route_distance_m = route_distance_km * 1000
                                avg_speed_kmh = np.mean([f.get('speed_kmh',40) for f in st.session_state.fleet]) if st.session_state.fleet else 40
                                route_duration_s = (route_distance_km / max(avg_speed_kmh,0.1)) * 3600
                                ls = {"type":"Feature","properties":{"vehicle":r['vehicle_tipo']},"geometry":{"type":"LineString","coordinates":coords}}
                                features.append(ls)

                            per_route_metrics.append({
                                'vehicle_idx': r.get('vehicle_idx'),
                                'vehicle_tipo': r.get('vehicle_tipo'),
                                'distance_km': round(route_distance_m/1000, 3),
                                'duration_min': round(route_duration_s/60, 1),
                                'cost': round((route_distance_m/1000) * cost_per_km, 2)
                            })
                            total_distance_m += route_distance_m
                            total_duration_s += route_duration_s

                        if features:
                            fc = {"type":"FeatureCollection","features":features}
                            st.session_state.route_geojson = fc
                            # totals
                            if total_distance_m > 0:
                                st.session_state.route_metrics = {"distance_m": total_distance_m, "duration_s": total_duration_s, "per_route": per_route_metrics}
                            else:
                                # fallback to solver estimates (km -> m)
                                st.session_state.route_metrics = {"distance_m": metrics_est['distance_km']*1000, "duration_s": metrics_est['time_min']*60, "per_route": per_route_metrics}
                            st.session_state.last_fingerprint = fp
                            st.success("Rutas calculadas y visualizadas.")
                        else:
                            st.error("No se obtuvieron geometrías para las rutas.")

# --- MAIN LAYOUT / MAPA ---
col1, col2 = st.columns((2,1))
with col1:
    st.subheader("Mapa")
    m = folium.Map(location=st.session_state.map_center, zoom_start=11, tiles="OpenStreetMap")

    # Add base layer group for pedidos by priority
    priority_layer = folium.FeatureGroup(name="Pedidos (por prioridad)", show=True)
    # priority colors
    pr_colors = {'alta':'red','media':'orange','baja':'green'}
    for i,p in enumerate(st.session_state.puntos):
        pr = str(p.get('prioridad','Media')).strip().lower()
        color = pr_colors.get(pr, 'blue')
        popup_html = f"<b>{p.get('nombre')}</b><br>Prio: {p.get('prioridad')}<br>Peso: {p.get('peso')} kg<br>Vol: {p.get('volumen')} m3<br>TW: {p.get('tw_start')} - {p.get('tw_end')}<br>{p.get('direccion','')}"
        folium.CircleMarker(location=[p['lat'], p['lon']],
                            radius=6 + min(max(p.get('peso',0)/20, 0), 10),
                            color=color,
                            fill=True,
                            fill_opacity=0.9,
                            popup=popup_html,
                            tooltip=f"{p.get('nombre')} ({p.get('prioridad')})").add_to(priority_layer)
    priority_layer.add_to(m)

    # Center marker
    if st.session_state.centro:
        folium.Marker(location=st.session_state.centro, icon=folium.Icon(color='darkred', icon='home'), popup="Centro").add_to(m)

    # If we have route GeoJSON, add each feature in its own layer with distinct colors
    if st.session_state.route_geojson:
        # choose distinct colors
        colors = ['blue','purple','cadetblue','darkgreen','darkorange','black','pink','gray']
        # iterate features and add to separate groups
        for idx, feat in enumerate(st.session_state.route_geojson.get('features', [])):
            veh = feat.get('properties', {}).get('vehicle', f'Ruta {idx+1}')
            color = colors[idx % len(colors)]
            layer = folium.FeatureGroup(name=f"Ruta: {veh}", show=(idx==0))
            geom = feat.get('geometry', {})
            if geom and geom.get('type') == 'LineString':
                coords = geom.get('coordinates', [])
                # folium expects [lat, lon]
                polyline = [[c[1], c[0]] for c in coords]
                folium.PolyLine(locations=polyline, color=color, weight=4, opacity=0.8, popup=f"Veh: {veh}").add_to(layer)
                # add start/end markers
                folium.CircleMarker(location=polyline[0], radius=5, color=color, fill=True, fill_opacity=1, popup=f"{veh} inicio").add_to(layer)
                folium.CircleMarker(location=polyline[-1], radius=5, color=color, fill=True, fill_opacity=1, popup=f"{veh} fin").add_to(layer)
            else:
                # generic geometry fallback: draw as GeoJson
                folium.GeoJson(feat, style_function=lambda x, col=color: {"color":col,"weight":4}).add_to(layer)
            layer.add_to(m)

    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Render map
    map_data = st_folium(m, width='100%', height=650)

    # click handling: select center or (optionally) allow manual adding for pedidos without coords
    if map_data and map_data.get("last_clicked"):
        lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        if st.session_state.seleccionando_centro:
            st.session_state.centro = [lat, lon]
            st.session_state.seleccionando_centro = False
            st.success("Centro definido en el mapa.")
        else:
            # Manual ad-hoc add
            st.session_state.puntos.append({'nombre': f'Punto {len(st.session_state.puntos)+1}', 'peso':0.0,'volumen':0.0,'prioridad':'Media','tw_start':'08:00','tw_end':'18:00','direccion':'','lat':lat,'lon':lon,'service_time':5})
            st.success("Punto agregado manualmente.")

with col2:
    st.subheader("Estadísticas & KPIs")
    num = len(st.session_state.puntos)
    st.metric("Pedidos totales", num)
    # If route metrics exist, show consolidated KPIs and table
    if st.session_state.route_metrics:
        dist_km = st.session_state.route_metrics['distance_m']/1000
        dur_min = st.session_state.route_metrics['duration_s']/60
        st.metric("Distancia total (estimada)", f"{dist_km:.2f} km")
        st.metric("Tiempo total (estimado)", f"{dur_min:.0f} min")
        total_cost = dist_km * cost_per_km
        st.metric("Costo estimado (total)", f"{total_cost:.2f}")
        # show per-route table if available
        per_route = st.session_state.route_metrics.get('per_route', [])
        if per_route:
            df_per_route = pd.DataFrame(per_route)
            # add readable columns
            df_per_route = df_per_route[['vehicle_tipo','distance_km','duration_min','cost']].rename(columns={
                'vehicle_tipo':'Vehículo',
                'distance_km':'Distancia (km)',
                'duration_min':'Duración (min)',
                'cost':'Costo'
            })
            st.subheader("Métricas por ruta/vehículo")
            st.dataframe(df_per_route, use_container_width=True)
    else:
        st.info("Aún no se han calculado rutas. Usa 'Calcular rutas (VRPTW)' en la barra lateral.")

st.subheader("Detalle Pedidos")
st.dataframe(crear_tabla_de_pedidos(st.session_state.puntos), use_container_width=True)
