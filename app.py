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
if 'last_click' not in st.session_state:
    st.session_state.last_click = None
if 'last_fingerprint' not in st.session_state:
    st.session_state.last_fingerprint = None

# --- CONFIGURAR ORS ---
# Pega aquí tu API key ORS válida con permisos (Directions). Si no quieres usar ORS para polylines,
# deja vacía la variable y el app usará líneas rectas como fallback.
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImY5MTA5MmE2NzVmZDRhYjBhMTk4YjZiNWNiMWY2YjQzIiwiaCI6Im11cm11cjY0In0="  # <-- PON TU API KEY AQUÍ (o deja vacío para fallback sin ORS)
if ORS_API_KEY:
    client = openrouteservice.Client(key=ORS_API_KEY)
else:
    client = None

# --- HELPERS ---
def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia Haversine en km."""
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    R = 6371  # Earth radius km
    return R * c

def get_coords_from_city(city_name):
    try:
        geolocator = Nominatim(user_agent="routing_app")
        location = geolocator.geocode(city_name, timeout=10)
        if location:
            return [location.latitude, location.longitude]
    except Exception as e:
        st.warning(f"No se pudo geocodificar la ciudad: {e}")
    return None

def crear_tabla_de_pedidos(puntos):
    if not puntos:
        return pd.DataFrame(columns=[
            'ID Pedido', 'Destino', 'Prioridad', 'Peso (kg)', 'Volumen (m3)',
            'Service (min)', 'TW Inicio', 'TW Fin', 'Latitud', 'Longitud'
        ])
    data = {
        'ID Pedido': [f"Pedido {i+1}" for i in range(len(puntos))],
        'Destino': [p.get('nombre', f'Punto {i+1}') for i, p in enumerate(puntos)],
        'Prioridad': [p.get('prioridad', 'Media') for p in puntos],
        'Peso (kg)': [p.get('peso', 0.0) for p in puntos],
        'Volumen (m3)': [p.get('volumen', 0.0) for p in puntos],
        'Service (min)': [p.get('service_time', 5) for p in puntos],
        'TW Inicio': [p.get('tw_start','00:00') for p in puntos],
        'TW Fin': [p.get('tw_end','23:59') for p in puntos],
        'Latitud': [p['lat'] for p in puntos],
        'Longitud': [p['lon'] for p in puntos]
    }
    return pd.DataFrame(data)

def fingerprint(centro, puntos, fleet_cfg):
    obj = {
        'centro': centro,
        'puntos': [(p['lat'], p['lon'], p.get('prioridad','Media'), p.get('peso',0), p.get('volumen',0),
                    p.get('service_time',0), p.get('tw_start','00:00'), p.get('tw_end','23:59')) for p in puntos],
        'fleet': fleet_cfg
    }
    s = json.dumps(obj, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()

def time_str_to_minutes(tstr):
    """Convierte 'HH:MM' -> minutos desde medianoche."""
    try:
        h, m = map(int, tstr.split(':'))
        return h * 60 + m
    except Exception:
        return 0

def minutes_to_time_str(minutes):
    h = int(minutes // 60)
    m = int(minutes % 60)
    return f"{h:02d}:{m:02d}"

def solve_vrptw_depot_first(centro, puntos, fleet_cfg, avg_speed_kmph=40):
    """
    Resuelve una versión CVRPTW aproximada con OR-Tools usando matriz de tiempos basada en haversine.
    Retorna: list_of_routes (each is list of node indices with depot=0), metrics estimation, and coordinates per node.
    Node indexing: 0 => depot, 1..N => pedidos
    """
    # Construir lista de nodos (depot + pedidos)
    nodes = []
    # Depot
    nodes.append({'lat': centro[0], 'lon': centro[1], 'demand_w': 0, 'demand_v': 0,
                  'service': 0, 'tw_start': fleet_cfg['shift_start'], 'tw_end': fleet_cfg['shift_end']})
    # Pedidos
    for p in puntos:
        nodes.append({
            'lat': p['lat'],
            'lon': p['lon'],
            'demand_w': p.get('peso', 0.0),
            'demand_v': p.get('volumen', 0.0),
            'service': p.get('service_time', 5),
            'tw_start': p.get('tw_start', '00:00'),
            'tw_end': p.get('tw_end', '23:59')
        })

    N = len(nodes)
    # Matriz de tiempos (minutos) y distancias (km)
    time_matrix = [[0]*N for _ in range(N)]
    dist_matrix = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                time_matrix[i][j] = 0
                dist_matrix[i][j] = 0
            else:
                km = haversine_km(nodes[i]['lat'], nodes[i]['lon'], nodes[j]['lat'], nodes[j]['lon'])
                # tiempo en minutos usando velocidad promedio
                travel_time_min = (km / max(avg_speed_kmph, 1)) * 60.0
                time_matrix[i][j] = int(round(travel_time_min))
                dist_matrix[i][j] = km

    # Datos para OR-Tools
    demands_w = [int(round(n['demand_w'])) for n in nodes]  # kg
    demands_v = [int(round(n['demand_v']*1000)) for n in nodes]  # transformar m3 a litros approximado (x1000) para entero
    service_times = [int(round(n['service'])) for n in nodes]
    time_windows = [(time_str_to_minutes(n['tw_start']), time_str_to_minutes(n['tw_end'])) for n in nodes]

    num_vehicles = fleet_cfg['num_vehicles']
    vehicle_capacity_w = int(round(fleet_cfg['capacity_weight']))  # kg
    vehicle_capacity_v = int(round(fleet_cfg['capacity_volume']*1000))  # same scale as demands_v
    depot_index = 0

    # Crear manager y modelo
    manager = pywrapcp.RoutingIndexManager(len(time_matrix), num_vehicles, depot_index)
    routing = pywrapcp.RoutingModel(manager)

    # Transito callback (tiempo en minutos)
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return time_matrix[from_node][to_node] + service_times[to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Agregar dimensión de tiempo
    time = 'Time'
    routing.AddDimension(
        transit_callback_index,
        60*24,  # slack / maximum waiting (large)
        60*24,  # maximum time per vehicle
        False,  # don't force start cumul to zero
        time
    )
    time_dimension = routing.GetDimensionOrDie(time)

    # Añadir ventanas horarias
    for node_idx in range(len(time_matrix)):
        index = manager.NodeToIndex(node_idx)
        start, end = time_windows[node_idx]
        time_dimension.CumulVar(index).SetRange(start, end)

    # Forzar ventana inicial a start of shift for depot nodes per vehicle start
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        ts = time_str_to_minutes(fleet_cfg['shift_start'])
        te = time_str_to_minutes(fleet_cfg['shift_end'])
        time_dimension.CumulVar(index).SetRange(ts, te)

    # Capacities (weight)
    def demand_w_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return demands_w[from_node]
    demand_w_idx = routing.RegisterUnaryTransitCallback(demand_w_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_w_idx,
        0,  # null capacity slack
        [vehicle_capacity_w]*num_vehicles,  # vehicle capacities
        True,  # start cumul to zero
        'CapacityW'
    )

    # Capacities (volume)
    def demand_v_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return demands_v[from_node]
    demand_v_idx = routing.RegisterUnaryTransitCallback(demand_v_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_v_idx,
        0,
        [vehicle_capacity_v]*num_vehicles,
        True,
        'CapacityV'
    )

    # Penalizar visitas no serviced si necesario (big penalty) - allow drop? We'll not allow drops here (could allow if needed)
    # sin drops: for large problemas puede ser imposible; en ese caso OR-Tools no encontrará solución factible.

    # Parámetros de búsqueda
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 20  # límite de búsqueda (ajustable)
    search_parameters.log_search = False

    # Resolver
    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        st.error("No se encontró solución factible con las restricciones dadas. Prueba relajar ventanas o aumentar vehículos/capacidad.")
        return None, None, None

    # Extraer rutas
    routes = []
    estimated_total_distance_km = 0.0
    estimated_total_time_min = 0.0
    for v in range(num_vehicles):
        idx = routing.Start(v)
        route_nodes = []
        route_dist_km = 0.0
        route_time_min = 0.0
        if routing.IsEnd(solution.Value(routing.NextVar(idx))):
            # vehículo sin asignaciones
            routes.append([])
            continue
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            route_nodes.append(node)
            next_idx = solution.Value(routing.NextVar(idx))
            if routing.IsEnd(next_idx):
                # end node
                break
            next_node = manager.IndexToNode(next_idx)
            route_dist_km += dist_matrix[node][next_node]
            route_time_min += time_matrix[node][next_node] + service_times[next_node]
            idx = next_idx
        # Append also end node if desired (depot end)
        routes.append(route_nodes)
        estimated_total_distance_km += route_dist_km
        estimated_total_time_min += route_time_min

    # Convert node indices to coordinates sequences for each route (start depot + visits + depot end optional)
    coords_per_route = []
    for r in routes:
        if not r:
            coords_per_route.append([])
            continue
        seq = []
        # start at depot
        seq.append([nodes[0]['lon'], nodes[0]['lat']])
        # add visited nodes in order (exclude depot if present as repeated)
        for node_idx in r:
            if node_idx == 0:
                continue
            seq.append([nodes[node_idx]['lon'], nodes[node_idx]['lat']])
        # optionally return to depot depending on policy; we include return for route drawing:
        seq.append([nodes[0]['lon'], nodes[0]['lat']])
        coords_per_route.append(seq)

    metrics_est = {'distance_km': estimated_total_distance_km, 'time_min': estimated_total_time_min}

    return routes, coords_per_route, metrics_est

def request_ors_route(coords_sequence):
    """Pide a ORS la ruta por calles para la secuencia dada (lista [lon,lat]). Devuelve geojson y metrics."""
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
    except openrouteservice.exceptions.ApiError as e:
        st.error(f"ORS API Error: {e}")
    except Exception as e:
        st.warning(f"Error solicitando ORS route: {e}")
    return None, None

# --- UI ---

st.title("🗺️ Panel de Control para Optimización de Rutas")
st.write("Haz clic en el mapa para agregar pedidos. Define centro, prioridades, ventanas horarias y capacidades de vehículo. Luego presiona 'Calcular Ruta (VRPTW)'.")
st.write("")  # espacio

# Sidebar inputs
with st.sidebar:
    st.header("⚙️ Configuración General")
    ciudad = st.text_input("📍 Ciudad de operación", "Bogotá")
    if st.button("Buscar Ciudad"):
        new = get_coords_from_city(ciudad)
        if new:
            st.session_state.map_center = new
            st.success(f"Mapa centrado en {ciudad}.")

    st.divider()
    st.subheader("Centro de Distribución")
    if st.button("📍 Seleccionar Centro en el Mapa"):
        st.session_state.seleccionando_centro = True
        st.info("Haz clic en el mapa para definir el centro.")

    st.divider()
    st.subheader("Pedidos (editar campos por pedido)")
    st.caption("Para cada pedido ajusta nombre, prioridad, peso (kg), volumen (m³), tiempo de servicio (min) y ventana horaria.")

    # Mostrar campos editables para cada punto
    for i, punto in enumerate(st.session_state.puntos):
        st.markdown(f"**Punto {i+1}**")
        punto['nombre'] = st.text_input(f"Nombre {i+1}", value=punto.get('nombre', f'Punto {i+1}'), key=f"nombre_{i}")
        punto['prioridad'] = st.selectbox("Prioridad", ('Baja','Media','Alta','Muy Alta'),
                                          index=['Baja','Media','Alta','Muy Alta'].index(punto.get('prioridad','Media')),
                                          key=f"prio_{i}")
        punto['peso'] = st.number_input(f"Peso (kg) {i+1}", min_value=0.0, value=float(punto.get('peso', 0.0)), key=f"peso_{i}")
        punto['volumen'] = st.number_input(f"Volumen (m³) {i+1}", min_value=0.0, value=float(punto.get('volumen', 0.0)), key=f"vol_{i}")
        punto['service_time'] = st.number_input(f"Service time (min) {i+1}", min_value=0, value=int(punto.get('service_time', 5)), key=f"serv_{i}")
        # Time window inputs as text "HH:MM"
        tws = punto.get('tw_start', '08:00')
        twf = punto.get('tw_end', '18:00')
        tw_start = st.text_input(f"TW inicio {i+1} (HH:MM)", value=tws, key=f"twstart_{i}")
        tw_end = st.text_input(f"TW fin {i+1} (HH:MM)", value=twf, key=f"twend_{i}")
        punto['tw_start'] = tw_start
        punto['tw_end'] = tw_end
        st.write("---")

    if st.button("🗑️ Limpiar todos los puntos"):
        st.session_state.puntos = []
        st.session_state.route_geojson = None
        st.session_state.route_metrics = None

    st.divider()
    st.subheader("Configuración de Flota")
    num_vehicles = st.number_input("Número de vehículos", min_value=1, max_value=20, value=3)
    capacity_weight = st.number_input("Capacidad por vehículo (kg)", min_value=1.0, value=1000.0)
    capacity_volume = st.number_input("Capacidad por vehículo (m³)", min_value=0.1, value=8.0)
    shift_start = st.text_input("Shift start (HH:MM)", value="07:00")
    shift_end = st.text_input("Shift end (HH:MM)", value="19:00")
    avg_speed_kmph = st.number_input("Velocidad media (km/h) usada para estimación", min_value=10.0, value=40.0)

    fleet_cfg = {
        'num_vehicles': int(num_vehicles),
        'capacity_weight': float(capacity_weight),
        'capacity_volume': float(capacity_volume),
        'shift_start': shift_start,
        'shift_end': shift_end
    }

    st.divider()
    st.subheader("Control de Cálculo")
    if st.button("🚀 Calcular Ruta (VRPTW)"):
        # Validation
        if not st.session_state.centro:
            st.warning("Primero define el centro de distribución (selecciona en el mapa).")
        elif not st.session_state.puntos:
            st.warning("Agrega al menos un pedido en el mapa.")
        else:
            # fingerprint
            fp = fingerprint(st.session_state.centro, st.session_state.puntos, fleet_cfg)
            if fp == st.session_state.last_fingerprint and st.session_state.route_geojson:
                st.info("La ruta ya está calculada y actualizada.")
            else:
                with st.spinner("Resolviendo VRPTW (OR-Tools)..."):
                    result = solve_vrptw_depot_first(
                        st.session_state.centro,
                        st.session_state.puntos,
                        fleet_cfg,
                        avg_speed_kmph
                    )

                    # Validar que haya solución
                    if not result or result[0] is None or result[1] is None:
                        st.error("No se obtuvo una solución factible. Intenta relajar las ventanas horarias, aumentar vehículos o capacidad.")
                    else:
                        routes_idx, coords_per_route, metrics_est = result

                        # Para cada ruta devuelta, pedimos ORS o usamos fallback
                        combined_features = []
                        total_distance_m = 0
                        total_duration_s = 0
                    
                        for seq in coords_per_route:
                            if not seq:
                                continue
                            # seq es lista de [lon,lat] comenzando y terminando en depot
                            if client:
                                feat, m = request_ors_route(seq)
                                if feat:
                                    combined_features.append(feat)
                                    total_distance_m += m['distance_m']
                                    total_duration_s += m['duration_s']
                                else:
                                    # fallback: crear geojson LineString simple
                                    ls = {
                                        "type": "Feature",
                                        "properties": {},
                                        "geometry": {"type": "LineString", "coordinates": seq}
                                    }
                                    combined_features.append(ls)
                            else:
                                ls = {
                                    "type": "Feature",
                                    "properties": {},
                                    "geometry": {"type": "LineString", "coordinates": seq}
                                }
                                combined_features.append(ls)
                        # Guardar resultado como FeatureCollection
                        if combined_features:
                            fc = {"type": "FeatureCollection", "features": combined_features}
                            st.session_state.route_geojson = fc
                            if total_distance_m > 0:
                                st.session_state.route_metrics = {"distance_m": total_distance_m, "duration_s": total_duration_s}
                            else:
                                # usar estimaciones
                                st.session_state.route_metrics = {"distance_m": metrics_est['distance_km']*1000, "duration_s": metrics_est['time_min']*60}
                            st.session_state.last_fingerprint = fp
                            st.success("Ruta calculada y visualizada.")
                        else:
                            st.error("No se pudo generar geometría de ruta (ni ORS ni fallback).")

# --- MAIN LAYOUT ---
col1, col2 = st.columns((2,1))

with col1:
    st.subheader("Mapa")
    m = folium.Map(location=st.session_state.map_center, zoom_start=13)

    # Mostrar centro
    if st.session_state.centro:
        folium.Marker(st.session_state.centro, icon=folium.Icon(color='red', icon='home'), popup="Centro").add_to(m)

    # Mostrar pedidos
    for i, punto in enumerate(st.session_state.puntos):
        folium.Marker([punto['lat'], punto['lon']],
                      popup=f"{punto.get('nombre','')}\n{punto.get('prioridad','')}\nPeso:{punto.get('peso',0)}kg\nTW: {punto.get('tw_start','')} - {punto.get('tw_end','')}",
                      tooltip=f"Punto {i+1}").add_to(m)

    # Dibujar ruta si existe
    if st.session_state.route_geojson:
        folium.GeoJson(st.session_state.route_geojson, name='Ruta', style_function=lambda x: {"color":"blue","weight":4,"opacity":0.8}).add_to(m)
    else:
        # guía visual con líneas gris si hay centro
        if st.session_state.centro:
            for punto in st.session_state.puntos:
                folium.PolyLine([st.session_state.centro, [punto['lat'], punto['lon']]], color='gray', weight=1, dash_array='5').add_to(m)

    # Render map and capture clicks
    map_data = st_folium(m, width='100%', height=600)

    # Capture user clicks (center selection or add a point)
    if map_data and map_data.get("last_clicked"):
        clicked = map_data["last_clicked"]
        lat, lon = clicked["lat"], clicked["lng"]
        # avoid duplicate same click
        if st.session_state.last_click and abs(st.session_state.last_click[0]-lat) < 1e-9 and abs(st.session_state.last_click[1]-lon) < 1e-9:
            pass
        else:
            st.session_state.last_click = (lat, lon)
            if st.session_state.seleccionando_centro:
                st.session_state.centro = [lat, lon]
                st.session_state.seleccionando_centro = False
                st.session_state.route_geojson = None
                st.session_state.route_metrics = None
                st.success("Centro definido.")
            else:
                # Agregar nuevo pedido con valores por defecto
                if len(st.session_state.puntos) >= 200:
                    st.warning("Máximo de pedidos alcanzado (200).")
                else:
                    st.session_state.puntos.append({
                        "lat": lat,
                        "lon": lon,
                        "prioridad": "Media",
                        "nombre": f"Punto {len(st.session_state.puntos)+1}",
                        "peso": 0.0,
                        "volumen": 0.0,
                        "service_time": 5,
                        "tw_start": "08:00",
                        "tw_end": "18:00"
                    })
                    st.session_state.route_geojson = None
                    st.session_state.route_metrics = None
                    st.success(f"Punto {len(st.session_state.puntos)} agregado.")

with col2:
    st.subheader("Estadísticas y KPIs")
    num = len(st.session_state.puntos)
    st.metric("Pedidos Totales", num)
    if st.session_state.route_metrics:
        dist_km = st.session_state.route_metrics['distance_m'] / 1000
        dur_min = st.session_state.route_metrics['duration_s'] / 60
        st.metric("Distancia (real)", f"{dist_km:.2f} km")
        st.metric("Tiempo (real)", f"{dur_min:.0f} min")
    else:
        # fallback estimado
        est_d = 5 + num * np.random.uniform(1.5, 3.5)
        est_t = num * (8 + np.random.uniform(-2,2))
        st.metric("Distancia (estimada)", f"{est_d:.2f} km")
        st.metric("Tiempo (estimado)", f"{est_t:.0f} min")

st.subheader("📄 Detalles de Pedidos")
df = crear_tabla_de_pedidos(st.session_state.puntos)
st.dataframe(df, use_container_width=True)

