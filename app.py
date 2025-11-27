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
import io
import re

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
ORS_API_KEY = ""  # deja vacía si no tienes
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

def geocode_address(address, city_hint=None, retries=2, delay=1.0):
    if not address or str(address).strip() == '':
        return None, None
    geolocator = Nominatim(user_agent="routing_app")
    q = address if pd.isna(city_hint) else f"{address}, {city_hint}"
    for _ in range(retries):
        try:
            loc = geolocator.geocode(q, timeout=10)
            if loc:
                return loc.latitude, loc.longitude
        except Exception:
            time.sleep(delay)
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

# Re-usable parsing of a dataframe that looks like a fleet table
def parse_fleet_df(df):
    """
    Devuelve una lista de dicts: [{'tipo','cantidad','capacity_kg','capacity_m3','speed_kmh','shift_start','shift_end'}, ...]
    Maneja nombres de columna flexibles.
    """
    if df is None or df.shape[0] == 0:
        return []
    # Normalize columns
    cols_map = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=cols_map)
    out = []
    # column name candidates
    tipo_cols = [c for c in df.columns if any(x in c for x in ['tipo','vehiculo','vehicle','modelo'])]
    cantidad_cols = [c for c in df.columns if 'cant' in c or 'cantidad' in c]
    cap_kg_cols = [c for c in df.columns if 'capacidad' in c and 'kg' in c or 'kg' in c and 'cap' in c]
    cap_kg_cols += [c for c in df.columns if re.search(r'\bcapacidad_?kg\b', c)]
    cap_m3_cols = [c for c in df.columns if 'm3' in c or 'capacidad' in c and 'm3' in c]
    speed_cols = [c for c in df.columns if 'veloc' in c or 'km' in c and 'h' in c or 'velocidad' in c]
    shift_start_cols = [c for c in df.columns if 'turno' in c and 'inicio' in c or 'start' in c]
    shift_end_cols = [c for c in df.columns if 'turno' in c and 'fin' in c or 'end' in c]
    # Fallbacks
    if not tipo_cols and 'tipo_vehiculo' in df.columns:
        tipo_cols = ['tipo_vehiculo']
    for idx, row in df.iterrows():
        tipo = row[tipo_cols[0]] if tipo_cols else (row.iloc[0] if len(row)>0 else f"vehiculo_{idx+1}")
        cantidad = int(float(row[cantidad_cols[0]])) if cantidad_cols and pd.notna(row[cantidad_cols[0]]) else None
        # helper to parse float from mixed strings
        def pfloat(x, default=0.0):
            try:
                if pd.isna(x):
                    return default
                s = str(x).strip().replace(',','.')
                return float(re.findall(r'[-+]?\d*\.?\d+', s)[0]) if re.findall(r'[-+]?\d*\.?\d+', s) else default
            except:
                return default
        capkg = pfloat(row[cap_kg_cols[0]]) if cap_kg_cols else 0.0
        capm3 = pfloat(row[cap_m3_cols[0]]) if cap_m3_cols else 0.0
        speed = pfloat(row[speed_cols[0]]) if speed_cols else 40.0
        shift_s = str(row[shift_start_cols[0]]) if shift_start_cols and pd.notna(row[shift_start_cols[0]]) else '07:00'
        shift_e = str(row[shift_end_cols[0]]) if shift_end_cols and pd.notna(row[shift_end_cols[0]]) else '19:00'
        entry = {
            'tipo': str(tipo).strip(),
            'cantidad': cantidad if cantidad is not None else 1,
            'capacity_kg': capkg,
            'capacity_m3': capm3,
            'speed_kmh': speed,
            'shift_start': shift_s,
            'shift_end': shift_e
        }
        out.append(entry)
    return out

# Find pedidos df in sheets: looks for columns like 'nombre' and 'peso'
def find_pedidos_and_fleet_from_sheets(xls_dict):
    """
    xls_dict: dict of sheetname->DataFrame
    returns: cedi_info (may be {}), fleet_list (list), pedidos_df (DataFrame or None)
    """
    pedidos_df = None
    fleet_list = []
    cedi_info = {}
    sheet_names = list(xls_dict.keys())
    # First pass: try find pedidos sheet by column names
    for name, df in xls_dict.items():
        cols = [str(c).strip().lower() for c in df.columns]
        cols_join = " ".join(cols)
        if any('nombre' in c for c in cols) and any('peso' in c for c in cols):
            pedidos_df = df.copy()
            break
    # second: find fleet-like sheet
    for name, df in xls_dict.items():
        cols = [str(c).strip().lower() for c in df.columns]
        cols_join = " ".join(cols)
        if any(k in cols_join for k in ['tipo','vehiculo','capacidad','capacidad_kg','velocidad','velocidad_kmh']):
            # parse fleet
            fleet_list = parse_fleet_df(df.copy())
            break
    # If we didn't find pedidos_df, fallback to parse_single_sheet_excel on first sheet content
    if pedidos_df is None:
        # try to detect single-sheet format where sections are in one sheet
        first = xls_dict[sheet_names[0]]
        try:
            cedi_info_p, fleet_p, pedidos_p = parse_single_sheet_excel(first.fillna('').astype(str))
            if pedidos_p is not None and (len(pedidos_p.columns) > 0):
                cedi_info = cedi_info_p
                if fleet_p:
                    fleet_list = fleet_p
                pedidos_df = pedidos_p
        except Exception:
            pedidos_df = None
    return cedi_info, fleet_list, pedidos_df

# --- existing parse_single_sheet_excel (kept for single-sheet format) ---
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
    for i in range(min(40, nrows)):
        rowstr = " ".join(df0.iloc[i].str.strip().str.lower().tolist())
        if 'cedi' in rowstr or 'cedi de distribucion' in rowstr or 'centro' in rowstr:
            # tomar la siguiente fila no vacía como info
            for j in range(i+1, min(i+8, nrows)):
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

# --- VRPTW solver (OR-Tools) simplified (kept unchanged) ---
def solve_vrptw(centro, puntos, fleet_list, time_limit_seconds=20):
    if not centro or not puntos or not fleet_list:
        return None, None

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

    demands_w = [int(round(n['demand_w'])) for n in nodes]
    demands_v = [int(round(n['demand_v']*1000)) for n in nodes]
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
    vehicle_cap_v = [int(round(f['capacity_m3']*1000)) for f in fleet_list]
    depot_index = 0

    manager = pywrapcp.RoutingIndexManager(N, num_vehicles, depot_index)
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return time_matrix[from_node][to_node] + service_times[to_node]
    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    time_dim_name = 'Time'
    routing.AddDimension(
        transit_callback_index,
        24*60,
        24*60,
        False,
        time_dim_name
    )
    time_dimension = routing.GetDimensionOrDie(time_dim_name)

    for node_idx in range(N):
        index = manager.NodeToIndex(node_idx)
        start, end = time_windows[node_idx]
        time_dimension.CumulVar(index).SetRange(start, end)

    for v in range(num_vehicles):
        start_idx = routing.Start(v)
        shift_s = time_str_to_minutes(fleet_list[v].get('shift_start','07:00'))
        shift_e = time_str_to_minutes(fleet_list[v].get('shift_end','19:00'))
        time_dimension.CumulVar(start_idx).SetRange(shift_s, shift_e)

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

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = time_limit_seconds
    search_parameters.log_search = False

    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        return None, None

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
st.title("🗺️ Panel de Control para Optimización de Rutas (Formato múltiple Excel)")
st.write("Sube el archivo Excel (puede ser con varias hojas o una sola hoja con secciones).")

# Sidebar: cost input
with st.sidebar:
    st.header("Parámetros")
    cost_per_km = st.number_input("Costo por kilómetro (unidad monetaria)", min_value=0.0, value=0.5, step=0.1)
    st.write("Si quieres usar ORS activa tu API key en el código (si la tienes).")

with st.sidebar:
    st.header("📂 Cargar archivo (hojas múltiples o única)")
    uploaded = st.file_uploader("Selecciona .xlsx", type=['xlsx'])
    if uploaded:
        try:
            # Read all sheets (don't force header) but allow pandas to infer header
            xls = pd.read_excel(uploaded, sheet_name=None, dtype=str)
            st.write("Hojas detectadas:", list(xls.keys()))
            # Try to automatically find pedidos and fleet
            cedi_info, fleet_list, pedidos_df = find_pedidos_and_fleet_from_sheets(xls)
            # If pedidos_df None, also try to fallback to reading first sheet with header
            if pedidos_df is None:
                # try reading first sheet with header
                first_name = list(xls.keys())[0]
                tmp = pd.read_excel(uploaded, sheet_name=first_name, dtype=str)
                # if tmp has nombre/peso columns, use it
                cols = [str(c).strip().lower() for c in tmp.columns]
                if any('nombre' in c for c in cols) and any('peso' in c for c in cols):
                    pedidos_df = tmp.copy()
            # If fleet_list empty, try to find a fleet table in any sheet heuristically
            if (not fleet_list) and (len(xls) >= 2):
                # try second sheet
                second_name = list(xls.keys())[1]
                try:
                    fleet_list_candidate = parse_fleet_df(xls[second_name].copy())
                    if fleet_list_candidate:
                        fleet_list = fleet_list_candidate
                except Exception:
                    pass

            # If still nothing, as final fallback attempt parse_single_sheet_excel on first sheet (text mode)
            if (not pedidos_df or pedidos_df is None) and len(xls) >= 1:
                first_name = list(xls.keys())[0]
                df_text = pd.read_excel(uploaded, sheet_name=first_name, header=None, dtype=str)
                cedi_tmp, fleet_tmp, pedidos_tmp = parse_single_sheet_excel(df_text)
                if pedidos_tmp is not None and len(pedidos_tmp)>0:
                    pedidos_df = pedidos_tmp
                    if fleet_tmp:
                        fleet_list = fleet_tmp
                    if cedi_tmp:
                        cedi_info = cedi_tmp

            # Show results
            if cedi_info:
                st.markdown(f"**CEDI detectado:** {cedi_info.get('name','')} — {cedi_info.get('direccion','')}")
                if cedi_info.get('direccion'):
                    latc, lonc = geocode_address(cedi_info.get('direccion'), city_hint=None)
                    if latc:
                        st.session_state.map_center = [latc, lonc]
                        if not st.session_state.centro:
                            st.session_state.centro = [latc, lonc]

            if fleet_list:
                # store fleet list into session (standardized)
                st.session_state.fleet = fleet_list
                st.markdown("**Flota detectada:**")
                # show dataframe representation
                df_show = pd.DataFrame(fleet_list)
                st.dataframe(df_show)
            else:
                st.info("No se detectó tabla de flota automáticamente. Asegúrate de que la hoja de flota contenga columnas como 'tipo', 'capacidad_kg', 'capacidad_m3', 'velocidad_kmh' o 'turno_inicio'.")

            if pedidos_df is not None:
                st.markdown("**Pedidos detectados (preview)**")
                # Normalize columns to lowercase
                pedidos_df.columns = [str(c).strip().lower() for c in pedidos_df.columns]
                st.dataframe(pedidos_df.head(40))

                # Normalize and extract lat/lon, peso, volumen, etc.
                loaded = 0
                new_points = []
                for i, row in pedidos_df.iterrows():
                    # read fields safely with flexible column names
                    def col_get(dfrow, candidates, default=None):
                        for c in candidates:
                            if c in dfrow.index and pd.notna(dfrow[c]) and str(dfrow[c]).strip()!='':
                                return dfrow[c]
                        return default

                    nombre = col_get(row, ['nombre_pedido','nombre','pedido','pedido_nombre'], default=f'Pedido {i+1}')
                    peso_raw = col_get(row, ['peso','peso (kg)','peso_kg','kg'], default=0.0)
                    vol_raw = col_get(row, ['vol','vol (m³)','volumen','volumen (m3)','volumen_m3'], default=0.0)
                    prioridad = col_get(row, ['prioridad','prio','prioridad (1)'], default='Media')
                    tws = col_get(row, ['tw_start','twstart','tw_inicio','tw_inicio'], default='08:00')
                    twf = col_get(row, ['tw_end','twend','tw_fin','tw_fin'], default='18:00')
                    ciudad = col_get(row, ['ciudad','city'], default=None)
                    direccion = col_get(row, ['direccion','dirección','address'], default=None)
                    # lat/lon candidates
                    lat_val = col_get(row, ['lat','latitud','latitude'], default=None)
                    lon_val = col_get(row, ['lon','longitud','longitude','lng'], default=None)
                    # parse numeric with comma support
                    def to_float_safe(x, default=0.0):
                        try:
                            if pd.isna(x):
                                return default
                            s = str(x).strip().replace(',','.')
                            # remove non numeric trailing/leading
                            m = re.search(r'[-+]?\d*\.?\d+', s)
                            return float(m.group(0)) if m else default
                        except:
                            return default
                    peso = to_float_safe(peso_raw, 0.0)
                    volumen = to_float_safe(vol_raw, 0.0)

                    lat, lon = None, None
                    if lat_val is not None and lon_val is not None:
                        try:
                            lat = to_float_safe(lat_val, None)
                            lon = to_float_safe(lon_val, None)
                        except:
                            lat, lon = None, None

                    # if lat/lon not present try geocode using direccion+ciudad
                    if (lat is None or lon is None) and direccion:
                        lat, lon = geocode_address(str(direccion), city_hint=ciudad)
                        time.sleep(0.5)

                    if lat is not None and lon is not None:
                        new_points.append({
                            'nombre': str(nombre),
                            'peso': peso,
                            'volumen': volumen,
                            'prioridad': str(prioridad),
                            'tw_start': str(tws),
                            'tw_end': str(twf),
                            'direccion': direccion if direccion is not None else '',
                            'lat': float(lat),
                            'lon': float(lon),
                            'service_time': int(to_float_safe(col_get(row, ['service_time','service','servicio'], default=5), 5))
                        })
                        loaded += 1
                st.success(f"{loaded} pedidos con coordenadas cargados desde Excel.")
                st.session_state.puntos = new_points
            else:
                st.info("No se detectaron pedidos automáticamente. Revisa las hojas del Excel o el formato del archivo.")
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
            fp = fingerprint(st.session_state.centro, st.session_state.puntos, st.session_state.fleet)
            if fp == st.session_state.last_fingerprint and st.session_state.route_geojson:
                st.info("La ruta ya está calculada y actualizada.")
            else:
                with st.spinner("Resolviendo VRPTW (OR-Tools)..."):
                    routes_info, metrics_est = solve_vrptw(st.session_state.centro, st.session_state.puntos, st.session_state.fleet, time_limit_seconds=20)
                    if not routes_info:
                        st.error("No se encontró solución factible con las restricciones dadas.")
                    else:
                        features = []
                        per_route_metrics = []
                        total_distance_m = 0
                        total_duration_s = 0
                        for r in routes_info:
                            coords = r.get('coords', [])
                            if not coords:
                                continue
                            if client:
                                feat, m = request_ors_route(coords)
                                if feat and m:
                                    features.append(feat)
                                    route_distance_m = m['distance_m']
                                    route_duration_s = m['duration_s']
                                else:
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
                            else:
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
                            if total_distance_m > 0:
                                st.session_state.route_metrics = {"distance_m": total_distance_m, "duration_s": total_duration_s, "per_route": per_route_metrics}
                            else:
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

    priority_layer = folium.FeatureGroup(name="Pedidos (por prioridad)", show=True)
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

    if st.session_state.centro:
        folium.Marker(location=st.session_state.centro, icon=folium.Icon(color='darkred', icon='home'), popup="Centro").add_to(m)

    if st.session_state.route_geojson:
        colors = ['blue','purple','cadetblue','darkgreen','darkorange','black','pink','gray']
        for idx, feat in enumerate(st.session_state.route_geojson.get('features', [])):
            veh = feat.get('properties', {}).get('vehicle', f'Ruta {idx+1}')
            color = colors[idx % len(colors)]
            layer = folium.FeatureGroup(name=f"Ruta: {veh}", show=(idx==0))
            geom = feat.get('geometry', {})
            if geom and geom.get('type') == 'LineString':
                coords = geom.get('coordinates', [])
                polyline = [[c[1], c[0]] for c in coords]
                folium.PolyLine(locations=polyline, color=color, weight=4, opacity=0.8, popup=f"Veh: {veh}").add_to(layer)
                folium.CircleMarker(location=polyline[0], radius=5, color=color, fill=True, fill_opacity=1, popup=f"{veh} inicio").add_to(layer)
                folium.CircleMarker(location=polyline[-1], radius=5, color=color, fill=True, fill_opacity=1, popup=f"{veh} fin").add_to(layer)
            else:
                folium.GeoJson(feat, style_function=lambda x, col=color: {"color":col,"weight":4}).add_to(layer)
            layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    map_data = st_folium(m, width='100%', height=650)

    if map_data and map_data.get("last_clicked"):
        lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        if st.session_state.seleccionando_centro:
            st.session_state.centro = [lat, lon]
            st.session_state.seleccionando_centro = False
            st.success("Centro definido en el mapa.")
        else:
            st.session_state.puntos.append({'nombre': f'Punto {len(st.session_state.puntos)+1}', 'peso':0.0,'volumen':0.0,'prioridad':'Media','tw_start':'08:00','tw_end':'18:00','direccion':'','lat':lat,'lon':lon,'service_time':5})
            st.success("Punto agregado manualmente.")

with col2:
    st.subheader("Estadísticas & KPIs")
    num = len(st.session_state.puntos)
    st.metric("Pedidos totales", num)
    if st.session_state.route_metrics:
        dist_km = st.session_state.route_metrics['distance_m']/1000
        dur_min = st.session_state.route_metrics['duration_s']/60
        st.metric("Distancia total (estimada)", f"{dist_km:.2f} km")
        st.metric("Tiempo total (estimado)", f"{dur_min:.0f} min")
        total_cost = dist_km * cost_per_km
        st.metric("Costo estimado (total)", f"{total_cost:.2f}")
        per_route = st.session_state.route_metrics.get('per_route', [])
        if per_route:
            df_per_route = pd.DataFrame(per_route)
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
