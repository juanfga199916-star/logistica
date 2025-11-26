

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import openrouteservice
import hashlib
import json
from math import radians, cos, sin, asin, sqrt
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import time
from typing import List, Dict, Any, Tuple, Optional

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="Optimizador Logístico", page_icon="🚚", layout="wide")
ORS_API_KEY = ""  # <-- Pega aquí tu API Key de OpenRouteService si dispones de una; si no deja vacío.
ors_client = openrouteservice.Client(key=ORS_API_KEY) if ORS_API_KEY else None

# -------------------------
# Helpers
# -------------------------
def safe_float(val, default=0.0):
    try:
        if pd.isna(val):
            return default
        return float(str(val).replace(',', '.'))
    except Exception:
        return default

def time_str_to_minutes(tstr: Any) -> int:
    """Convert 'HH:MM' or 'H:MM' or numeric strings to minutes since midnight."""
    if tstr is None:
        return 0
    s = str(tstr).strip()
    if s == '':
        return 0
    try:
        parts = s.split(':')
        if len(parts) == 2:
            h = int(parts[0])
            m = int(parts[1])
            return h * 60 + m
        else:
            # maybe it's a float like 7.0 or '7'
            if '.' in s:
                s = s.split('.')[0]
            return int(s) * 60
    except Exception:
        return 0

def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in kilometers."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    R = 6371
    return R * c

def fingerprint(cedi, pedidos, fleet):
    obj = {'cedi': cedi, 'pedidos': [(p.get('lat'), p.get('lon'), p.get('peso'), p.get('volumen'), p.get('tw_start'), p.get('tw_end')) for p in pedidos], 'fleet': fleet}
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

# -------------------------
# DataManager: reads two-sheet excel, geocodes addresses
# -------------------------
class DataManager:
    def __init__(self, user_agent="route_app_geocode"):
        self.geolocator = Nominatim(user_agent=user_agent)
        self.geocode = RateLimiter(self.geolocator.geocode, min_delay_seconds=1.0)

    def read_excel_two_sheets(self, uploaded_file) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Expect two sheets:
         - 'Flota' with columns: tipo_vehiculo, capacidad_kg, capacidad_m3, velocidad_kmh, turno_inicio, turno_fin
         - 'Pedidos' with columns: nombre_pedido, peso, volumen, prioridad, tw_start, tw_end, Ciudad, Direccion
        Returns (df_flota, df_pedidos).
        """
        # read all sheets or specific sheets
        try:
            xls = pd.read_excel(uploaded_file, sheet_name=None)
        except Exception as e:
            raise RuntimeError(f"Error leyendo Excel: {e}")

        # Try common sheet names robustly
        sheet_names = {name.strip().lower(): name for name in xls.keys()}
        def get_sheet_by_names(possible_names):
            for nm in possible_names:
                if nm.lower() in sheet_names:
                    return xls[sheet_names[nm.lower()]]
            return None

        df_flota = get_sheet_by_names(['Flota', 'flota', 'Caracteristicas de flota', 'fleet'])
        df_pedidos = get_sheet_by_names(['Pedidos', 'pedidos', 'Caracteristicas de pedidos', 'orders'])

        if df_flota is None:
            raise RuntimeError("No se encontró la hoja 'Flota' en el archivo. Asegúrate que exista una hoja llamada 'Flota'.")
        if df_pedidos is None:
            raise RuntimeError("No se encontró la hoja 'Pedidos' en el archivo. Asegúrate que exista una hoja llamada 'Pedidos'.")

        return df_flota, df_pedidos

    def normalize_flota(self, df_flota: pd.DataFrame) -> List[Dict[str,Any]]:
        # normalize column names
        df = df_flota.copy()
        df.columns = [c.strip().lower() for c in df.columns]
        # required columns mapping
        # expected: tipo_vehiculo, capacidad_kg, capacidad_m3, velocidad_kmh, turno_inicio, turno_fin
        required = ['tipo_vehiculo', 'capacidad_kg', 'capacidad_m3', 'velocidad_kmh']
        # try to map if alternative names exist
        cols = df.columns.tolist()
        # ensure columns exist; if not, try to infer
        # Fill missing columns with defaults
        fleet_list = []
        for i, row in df.iterrows():
            tipo = row.get('tipo_vehiculo') or row.get('tipo') or row.get('vehicle_type') or f"vehiculo_{i+1}"
            cap_kg = safe_float(row.get('capacidad_kg') or row.get('capacidad') or row.get('capacity_kg'), 1000.0)
            cap_m3 = safe_float(row.get('capacidad_m3') or row.get('capacidad_m') or row.get('capacity_m3'), 8.0)
            speed = safe_float(row.get('velocidad_kmh') or row.get('velocidad') or row.get('speed_kmh'), 40.0)
            turno_i = str(row.get('turno_inicio') or row.get('turno_i') or row.get('shift_start') or '07:00')
            turno_f = str(row.get('turno_fin') or row.get('turno_f') or row.get('shift_end') or '19:00')
            fleet_list.append({'tipo': str(tipo).strip(), 'capacity_kg': cap_kg, 'capacity_m3': cap_m3, 'speed_kmh': speed, 'shift_start': turno_i, 'shift_end': turno_f})
        return fleet_list

    def normalize_pedidos(self, df_pedidos: pd.DataFrame) -> pd.DataFrame:
        df = df_pedidos.copy()
        df.columns = [c.strip().lower() for c in df.columns]
        # ensure expected columns exist by alias mapping
        # expected keys: nombre_pedido, peso, volumen, prioridad, tw_start, tw_end, ciudad, direccion
        # rename common variants
        col_map = {}
        for c in df.columns:
            lc = c.lower()
            if 'nombre' in lc and 'pedido' in lc:
                col_map[c] = 'nombre_pedido'
            elif lc == 'nombre':
                col_map[c] = 'nombre_pedido'
            elif 'peso' in lc:
                col_map[c] = 'peso'
            elif 'volumen' in lc or 'm3' in lc:
                col_map[c] = 'volumen'
            elif 'prioridad' in lc:
                col_map[c] = 'prioridad'
            elif 'tw_start' in lc or ('inicio' in lc and 'tw' in lc) or ('inicio' in lc and 'turno' not in lc):
                col_map[c] = 'tw_start'
            elif 'tw_end' in lc or ('fin' in lc and 'tw' in lc) or ('fin' in lc and 'turno' not in lc):
                col_map[c] = 'tw_end'
            elif 'ciudad' in lc:
                col_map[c] = 'ciudad'
            elif 'direccion' in lc or 'dirección' in lc:
                col_map[c] = 'direccion'
            elif lc == 'lat':
                col_map[c] = 'lat'
            elif lc == 'lon' or lc == 'lng' or 'long' in lc:
                col_map[c] = 'lon'
            elif 'service' in lc:
                col_map[c] = 'service_time'
        df = df.rename(columns=col_map)
        # ensure cols exist
        defaults = {'nombre_pedido': None, 'peso': 0.0, 'volumen': 0.0, 'prioridad': 'Media', 'tw_start': '08:00', 'tw_end': '18:00', 'ciudad': None, 'direccion': None}
        for k,v in defaults.items():
            if k not in df.columns:
                df[k] = v
        # fill NA
        df = df.fillna('')
        return df

    def geocode_row(self, address: str, city: str) -> Tuple[Optional[float], Optional[float]]:
        """Geocode address + city, return lat, lon or (None, None)."""
        if not address:
            return None, None
        q = f"{address}, {city}" if city else address
        try:
            loc = self.geocode(q)
            if loc:
                return loc.latitude, loc.longitude
        except Exception:
            # backoff and simple try
            try:
                time.sleep(1)
                loc = self.geocode(q)
                if loc:
                    return loc.latitude, loc.longitude
            except Exception:
                return None, None
        return None, None

    def pedidos_to_points(self, df_pedidos: pd.DataFrame, max_geocode=200) -> List[Dict[str,Any]]:
        df = self.normalize_pedidos(df_pedidos)
        points = []
        geocoded = 0
        for idx, row in df.iterrows():
            nombre = row.get('nombre_pedido') or f'Pedido {idx+1}'
            peso = safe_float(row.get('peso'), 0.0)
            volumen = safe_float(row.get('volumen'), 0.0)
            prioridad = row.get('prioridad') or 'Media'
            tws = str(row.get('tw_start') or '08:00')
            twf = str(row.get('tw_end') or '18:00')
            ciudad = row.get('ciudad') or ''
            direccion = row.get('direccion') or ''
            lat = None
            lon = None
            # If lat/lon present
            if 'lat' in df.columns and 'lon' in df.columns:
                try:
                    lat_val = row.get('lat')
                    lon_val = row.get('lon')
                    if str(lat_val).strip() != '' and str(lon_val).strip() != '':
                        lat = float(lat_val)
                        lon = float(lon_val)
                except Exception:
                    lat, lon = None, None
            # Geocode if no coords
            if (lat is None or lon is None) and direccion:
                if geocoded < max_geocode:
                    lat, lon = self.geocode_row(str(direccion), str(ciudad))
                    geocoded += 1
                    time.sleep(1)  # polite
            if lat is not None and lon is not None:
                points.append({
                    'nombre': str(nombre),
                    'peso': peso,
                    'volumen': volumen,
                    'prioridad': prioridad,
                    'tw_start': tws,
                    'tw_end': twf,
                    'ciudad': ciudad,
                    'direccion': direccion,
                    'lat': lat,
                    'lon': lon,
                    'service_time': int(safe_float(row.get('service_time', 5), 5))
                })
        return points

# -------------------------
# Routing Engine (OR-Tools VRPTW with capacity weight+volume)
# -------------------------
class RoutingEngine:
    def __init__(self):
        pass

    def solve_vrptw(self, cedi: Dict[str,float], pedidos: List[Dict[str,Any]], fleet: List[Dict[str,Any]], time_limit_seconds:int = 20) -> Tuple[Optional[List[Dict]], Optional[Dict]]:
        """
        cedi: {'lat':..., 'lon':...}
        pedidos: list of dicts with lat, lon, peso, volumen, tw_start, tw_end, service_time
        fleet: list of dicts with tipo, capacity_kg, capacity_m3, speed_kmh, shift_start, shift_end
        returns: (routes_info, metrics_est) or (None, None)
        """
        if not cedi or not pedidos or not fleet:
            return None, None

        # Nodes: depot (0) + pedidos 1..N
        nodes = [{'lat': cedi['lat'], 'lon': cedi['lon'], 'peso': 0.0, 'volumen': 0.0, 'service': 0, 'tw_start': '00:00', 'tw_end': '23:59'}]
        for p in pedidos:
            nodes.append({
                'lat': p['lat'], 'lon': p['lon'],
                'peso': safe_float(p.get('peso', 0.0), 0.0),
                'volumen': safe_float(p.get('volumen', 0.0), 0.0),
                'service': int(safe_float(p.get('service_time', 5), 5)),
                'tw_start': p.get('tw_start', '08:00'),
                'tw_end': p.get('tw_end', '18:00')
            })

        N = len(nodes)
        num_vehicles = len(fleet)

        # Build time and distance matrices (using average speed across fleet)
        avg_speed = np.mean([safe_float(f.get('speed_kmh', 40), 40) for f in fleet])
        time_matrix = [[0]*N for _ in range(N)]
        dist_matrix = [[0.0]*N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                km = haversine_km(nodes[i]['lat'], nodes[i]['lon'], nodes[j]['lat'], nodes[j]['lon'])
                dist_matrix[i][j] = km
                time_matrix[i][j] = int(round((km / max(avg_speed, 1.0)) * 60.0))

        # Demands & capacities (OR-Tools needs integers)
        demands_w = [int(round(n['peso'])) for n in nodes]
        demands_v = [int(round(n['volumen'] * 1000)) for n in nodes]  # scale m3 -> litres-ish
        service_times = [int(n['service']) for n in nodes]
        time_windows = [(time_str_to_minutes(n.get('tw_start','00:00')), time_str_to_minutes(n.get('tw_end','23:59'))) for n in nodes]

        vehicle_cap_w = [int(round(safe_float(f.get('capacity_kg', 1000), 1000))) for f in fleet]
        vehicle_cap_v = [int(round(safe_float(f.get('capacity_m3', 8), 8) * 1000)) for f in fleet]  # scaled

        # OR-Tools model
        manager = pywrapcp.RoutingIndexManager(N, num_vehicles, 0)
        routing = pywrapcp.RoutingModel(manager)

        # Transit (time) callback
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return time_matrix[from_node][to_node] + service_times[to_node]

        transit_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Time dimension
        routing.AddDimension(transit_callback_index, 24*60, 24*60, False, 'Time')
        time_dimension = routing.GetDimensionOrDie('Time')

        # Set time windows for each node
        for node_idx in range(N):
            index = manager.NodeToIndex(node_idx)
            tw_start, tw_end = time_windows[node_idx]
            time_dimension.CumulVar(index).SetRange(tw_start, tw_end)

        # Vehicle start windows from fleet shifts
        for v in range(num_vehicles):
            start_idx = routing.Start(v)
            shift_s = time_str_to_minutes(fleet[v].get('shift_start', '07:00'))
            shift_e = time_str_to_minutes(fleet[v].get('shift_end', '19:00'))
            time_dimension.CumulVar(start_idx).SetRange(shift_s, shift_e)

        # Capacity weight callback
        def demand_w_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return demands_w[from_node]
        demand_w_idx = routing.RegisterUnaryTransitCallback(demand_w_callback)
        routing.AddDimensionWithVehicleCapacity(demand_w_idx, 0, vehicle_cap_w, True, 'CapacityW')

        # Capacity volume callback
        def demand_v_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return demands_v[from_node]
        demand_v_idx = routing.RegisterUnaryTransitCallback(demand_v_callback)
        routing.AddDimensionWithVehicleCapacity(demand_v_idx, 0, vehicle_cap_v, True, 'CapacityV')

        # Search parameters
        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_params.time_limit.seconds = time_limit_seconds
        search_params.log_search = False

        solution = routing.SolveWithParameters(search_params)
        if solution is None:
            return None, None

        # Extract routes
        routes_info = []
        total_km = 0.0
        total_min = 0.0
        for v in range(num_vehicles):
            index = routing.Start(v)
            route_nodes = []
            route_coords = []
            route_km = 0.0
            route_min = 0.0
            # If vehicle has no assigned nodes, NextVar(start) == End?
            if routing.IsEnd(solution.Value(routing.NextVar(index))):
                # no assignments
                routes_info.append({'vehicle_id': v, 'vehicle_type': fleet[v].get('tipo', 'vehiculo'), 'coords': [], 'distance_km': 0.0, 'nodes': []})
                continue

            # start at depot
            route_coords.append([nodes[0]['lon'], nodes[0]['lat']])
            while not routing.IsEnd(index):
                node_idx = manager.IndexToNode(index)
                if node_idx != 0:
                    route_nodes.append(node_idx)
                    route_coords.append([nodes[node_idx]['lon'], nodes[node_idx]['lat']])
                prev_index = index
                index = solution.Value(routing.NextVar(index))
                next_node = manager.IndexToNode(index)
                # accumulate distances and times
                route_km += dist_matrix[node_idx][next_node] if node_idx < len(dist_matrix) and next_node < len(dist_matrix) else 0.0
                route_min += time_matrix[node_idx][next_node] if node_idx < len(time_matrix) and next_node < len(time_matrix) else 0
            # append depot end
            route_coords.append([nodes[0]['lon'], nodes[0]['lat']])

            total_km += route_km
            total_min += route_min

            routes_info.append({'vehicle_id': v, 'vehicle_type': fleet[v].get('tipo', 'vehiculo'), 'coords': route_coords, 'distance_km': route_km, 'nodes': route_nodes})

        metrics = {'distance_km': total_km, 'time_min': total_min}
        return routes_info, metrics

# -------------------------
# Utility: ORS route geometry fetch
# -------------------------
def get_ors_geojson_for_coords(coords: List[List[float]]):
    """coords: list of [lon, lat] pairs"""
    if not ors_client:
        return None
    try:
        # openrouteservice expects "coordinates" param in some clients or "coordinates" positional
        # using directions with geojson format
        route = ors_client.directions(coords, profile='driving-car', format='geojson')
        feat = route.get('features', [None])[0]
        return feat
    except Exception as e:
        st.warning(f"ORS directions failed: {e}")
        return None

# -------------------------
# UI (Main)
# -------------------------
def main():
    st.title("🚚 Optimizador Logístico — carga Excel (Flota + Pedidos)")

    dm = DataManager()
    engine = RoutingEngine()

    # Sidebar: CEDI manual input (option A)
    st.sidebar.header("CEDI (centro de distribución) — escribe y geocodifica")
    cedi_address = st.sidebar.text_input("Dirección del CEDI (ej. Cl. 9 #12-2)", value="")
    cedi_city = st.sidebar.text_input("Ciudad del CEDI (ej. Buga)", value="")
    if st.sidebar.button("Geocodificar CEDI"):
        if cedi_address.strip() == '':
            st.sidebar.error("Escribe la dirección del CEDI.")
        else:
            latc, lonc = dm.geocode_row(cedi_address, cedi_city)
            if latc is None:
                st.sidebar.error("No se pudo geocodificar el CEDI. Revisa la dirección.")
            else:
                st.session_state.cedi = {'lat': latc, 'lon': lonc, 'direccion': cedi_address, 'ciudad': cedi_city}
                st.sidebar.success(f"CEDI geocodificado: ({latc:.6f}, {lonc:.6f})")

    # Sidebar: Upload Excel
    st.sidebar.markdown("---")
    st.sidebar.header("Carga Excel (dos hojas obligatorias)")
    uploaded = st.sidebar.file_uploader("Sube archivo .xlsx con hojas 'Flota' y 'Pedidos'", type=['xlsx'])

    if uploaded:
        try:
            df_flota, df_pedidos = dm.read_excel_two_sheets(uploaded)
            fleet_list = dm.normalize_flota(df_flota)
            pedidos_points = dm.pedidos_to_points(df_pedidos)
            st.session_state.df_flota = df_flota
            st.session_state.df_pedidos = df_pedidos
            st.session_state.fleet = fleet_list
            st.session_state.pedidos = pedidos_points
            st.sidebar.success(f"Archivo cargado. Flota: {len(fleet_list)} tipos. Pedidos geocodificados: {len(pedidos_points)}")
            # if CEDI not defined, try to infer from sheet 'Flota' first row if exists: optional
        except Exception as e:
            st.sidebar.error(f"Error procesando archivo: {e}")

    # Allow manual edit of fleet in UI (optional)
    if 'fleet' in st.session_state and st.session_state.fleet:
        st.sidebar.markdown("**Flota detectada (puedes editar valores)**")
        # show small editable table (simplified): we will only display
        try:
            df_fleet_show = pd.DataFrame(st.session_state.fleet)
            st.sidebar.dataframe(df_fleet_show)
        except Exception:
            pass

    st.sidebar.markdown("---")
    # Calculation controls
    st.sidebar.header("Cálculo")
    calc_button = st.sidebar.button("🚀 Calcular rutas (VRPTW)")

    # Map and main output
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Mapa")
        # center map on CEDI if available else Bogotá default
        if 'cedi' in st.session_state and st.session_state.cedi:
            center = [st.session_state.cedi['lat'], st.session_state.cedi['lon']]
        else:
            center = [4.60971, -74.08175]

        m = folium.Map(location=center, zoom_start=11)

        # Show CEDI
        if 'cedi' in st.session_state and st.session_state.cedi:
            folium.Marker(
                [st.session_state.cedi['lat'], st.session_state.cedi['lon']],
                icon=folium.Icon(color='red', icon='home'),
                popup=f"CEDI: {st.session_state.cedi.get('direccion','')}, {st.session_state.cedi.get('ciudad','')}"
            ).add_to(m)

        # Show pedidos markers
        if 'pedidos' in st.session_state and st.session_state.pedidos:
            for i, p in enumerate(st.session_state.pedidos):
                popup = f"<b>{p.get('nombre')}</b><br>Peso: {p.get('peso')} kg<br>Vol: {p.get('volumen')} m3<br>TW: {p.get('tw_start')} - {p.get('tw_end')}<br>{p.get('direccion','')}"
                folium.CircleMarker([p['lat'], p['lon']], radius=4, color='blue', fill=True, popup=popup, tooltip=f"P{ i+1 }").add_to(m)

        # If routes exist in session state, display them
        if 'routes' in st.session_state and st.session_state.routes:
            colors_by_type = {}
            palette = ['green','orange','purple','blue','darkred','cadetblue','darkgreen','brown','pink']
            # assign color per vehicle type
            for r in st.session_state.routes:
                vtype = r.get('vehicle_type', 'vehiculo')
                if vtype not in colors_by_type:
                    colors_by_type[vtype] = palette[len(colors_by_type) % len(palette)]

            for r in st.session_state.routes:
                coords = r.get('coords', [])
                if not coords:
                    continue
                vtype = r.get('vehicle_type','vehiculo')
                color = colors_by_type.get(vtype, 'blue')
                # try ORS geometry for nicer polyline if ORS key set
                geo = None
                if ors_client:
                    try:
                        # coords must be list of [lon, lat]
                        geo = get_ors_geojson_for_coords(coords)
                    except Exception:
                        geo = None
                if geo:
                    folium.GeoJson(geo, style_function=lambda feat, col=color: {'color': col, 'weight': 4, 'opacity':0.8}).add_to(m)
                else:
                    # fallback: folium PolyLine expects [lat, lon] pairs
                    line_coords = [[lat, lon] for lon, lat in coords]
                    folium.PolyLine(line_coords, color=color, weight=4, opacity=0.8, tooltip=f"{vtype}").add_to(m)

        st_folium(m, width="100%", height=650)

    with col2:
        st.subheader("Métricas & Resultados")
        if 'routes' in st.session_state and st.session_state.routes:
            total_km = sum([r.get('distance_km', 0.0) for r in st.session_state.routes])
            total_veh = len([r for r in st.session_state.routes if r.get('coords')])
            st.metric("Distancia total estimada", f"{total_km:.2f} km")
            st.metric("Vehículos usados", total_veh)
            # cost per km example
            cost_per_km = st.number_input("Costo por km (unit)", min_value=0.0, value=0.5)
            st.metric("Costo estimado (total)", f"{(total_km * cost_per_km):.2f}")

            st.markdown("### Rutas por vehículo")
            for r in st.session_state.routes:
                if not r.get('coords'):
                    continue
                with st.expander(f"Vehículo {r['vehicle_id']+1} — {r['vehicle_type']}"):
                    st.write(f"Distancia (est): {r.get('distance_km',0):.2f} km")
                    st.write(f"Paradas: {len(r.get('nodes',[]))}")
                    st.write(f"Coords points: {len(r.get('coords',[]))}")
        else:
            st.info("Sube archivo con hojas 'Flota' y 'Pedidos', geocodifica CEDI y pulsa 'Calcular rutas (VRPTW)' en la barra lateral.")

    # -------------------------
    # Trigger calculation (after map shown for better UX)
    # -------------------------
    if calc_button:
        # validate
        if 'cedi' not in st.session_state or not st.session_state.cedi:
            st.sidebar.error("Debes geocodificar el CEDI en la barra lateral antes de calcular.")
        elif 'pedidos' not in st.session_state or not st.session_state.pedidos:
            st.sidebar.error("No hay pedidos (o no tienen coordenadas). Asegúrate de que la hoja 'Pedidos' tenga direcciones y/o lat/lon.")
        elif 'fleet' not in st.session_state or not st.session_state.fleet:
            st.sidebar.error("No hay flota definida (hoja 'Flota' vacía o con formato erróneo).")
        else:
            with st.spinner("Resolviendo VRPTW con OR-Tools..."):
                routes_info, metrics = engine.solve_vrptw(st.session_state.cedi, st.session_state.pedidos, st.session_state.fleet, time_limit_seconds=30)
                if routes_info is None:
                    st.sidebar.error("No se encontró una solución factible con las restricciones actuales. Intenta relajar ventanas/capacidad o aumentar vehículos.")
                else:
                    st.session_state.routes = routes_info
                    st.success(f"Encontradas {len([r for r in routes_info if r.get('coords')])} rutas.")
                    # compute ORS geometries optionally and metrics sums
                    total_distance_m = 0
                    total_duration_s = 0
                    features = []
                    for r in routes_info:
                        coords = r.get('coords', [])
                        if not coords:
                            continue
                        feat = None
                        m = None
                        if ors_client:
                            try:
                                feat = get_ors_geojson_for_coords(coords)
                                if feat:
                                    # try to get segment sums
                                    props = feat.get('properties',{})
                                    # fallback: not always present
                                    # request_ors route returns properties.segments with distance/duration values
                                    total_distance_m += sum(seg.get('distance',0) for seg in props.get('segments',[])) if props.get('segments') else 0
                                    total_duration_s += sum(seg.get('duration',0) for seg in props.get('segments',[])) if props.get('segments') else 0
                                    features.append(feat)
                                else:
                                    # fallback line
                                    ls = {"type":"Feature","properties":{"vehicle":r.get('vehicle_type')}, "geometry":{"type":"LineString","coordinates":coords}}
                                    features.append(ls)
                            except Exception:
                                ls = {"type":"Feature","properties":{"vehicle":r.get('vehicle_type')}, "geometry":{"type":"LineString","coordinates":coords}}
                                features.append(ls)
                        else:
                            ls = {"type":"Feature","properties":{"vehicle":r.get('vehicle_type')}, "geometry":{"type":"LineString","coordinates":coords}}
                            features.append(ls)

                    if features:
                        fc = {"type":"FeatureCollection","features":features}
                        st.session_state.route_geojson = fc

                    # Estimations if ORS not available
                    if total_distance_m == 0:
                        total_distance_km = metrics.get('distance_km', 0.0)
                        total_duration_min = metrics.get('time_min', 0.0)
                        st.session_state.route_metrics = {'distance_m': total_distance_km * 1000, 'duration_s': total_duration_min * 60}
                    else:
                        st.session_state.route_metrics = {'distance_m': total_distance_m, 'duration_s': total_duration_s}

    # Footer: show loaded tables
    st.markdown("---")
    st.subheader("Tablas cargadas")
    if 'df_flota' in st.session_state:
        st.markdown("**Flota (raw)**")
        st.dataframe(st.session_state.df_flota)
    if 'df_pedidos' in st.session_state:
        st.markdown("**Pedidos (raw, primeras 20 filas)**")
        st.dataframe(st.session_state.df_pedidos.head(20))

if __name__ == "__main__":
    main()
