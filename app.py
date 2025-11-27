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

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Panel de Control de Rutas", page_icon="🚚", layout="wide")

# --- ESTADO INICIAL ---
if 'puntos' not in st.session_state:
    st.session_state.puntos = []
if 'map_center' not in st.session_state:
    st.session_state.map_center = [4.60971, -74.08175]
if 'centro' not in st.session_state:
    st.session_state.centro = None
if 'route_geojson' not in st.session_state:
    st.session_state.route_geojson = None
if 'route_metrics' not in st.session_state:
    st.session_state.route_metrics = None

# --- CONFIGURAR ORS ---
ORS_API_KEY = "TU_API_KEY_AQUI"
client = openrouteservice.Client(key=ORS_API_KEY) if ORS_API_KEY else None

# --- FUNCIONES AUXILIARES ---
def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia Haversine en km."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 6371 * 2 * asin(sqrt(a))

def time_str_to_minutes(t):
    try:
        h, m = map(int, t.split(':'))
        return h*60 + m
    except:
        return 0

def solve_vrptw(centro, puntos, fleet_cfg, avg_speed_kmph=40):
    """Resuelve VRPTW aproximado."""
    nodes = [{'lat': centro[0], 'lon': centro[1], 'demand': 0, 'service': 0, 'tw_start': fleet_cfg['inicio_turno'], 'tw_end': fleet_cfg['fin_turno']}]
    for p in puntos:
        nodes.append({
            'lat': p['lat'], 'lon': p['lon'], 'demand': p['peso'], 'service': p.get('service_time', 5),
            'tw_start': p.get('tw_start', '08:00'), 'tw_end': p.get('tw_end', '18:00')
        })

    N = len(nodes)
    time_matrix = [[0]*N for _ in range(N)]
    dist_matrix = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j:
                km = haversine_km(nodes[i]['lat'], nodes[i]['lon'], nodes[j]['lat'], nodes[j]['lon'])
                dist_matrix[i][j] = km
                time_matrix[i][j] = int((km / avg_speed_kmph) * 60)

    manager = pywrapcp.RoutingIndexManager(N, fleet_cfg['num_vehiculos'], 0)
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_idx, to_idx):
        f, t = manager.IndexToNode(from_idx), manager.IndexToNode(to_idx)
        return time_matrix[f][t] + nodes[t]['service']
    transit_idx = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    routing.AddDimension(transit_idx, 60*24, 60*24, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    for node_idx in range(N):
        idx = manager.NodeToIndex(node_idx)
        start, end = time_str_to_minutes(nodes[node_idx]['tw_start']), time_str_to_minutes(nodes[node_idx]['tw_end'])
        time_dim.CumulVar(idx).SetRange(start, end)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.time_limit.seconds = 15

    sol = routing.SolveWithParameters(params)
    if not sol:
        return None, None

    coords_per_route = []
    for v in range(fleet_cfg['num_vehiculos']):
        idx = routing.Start(v)
        seq = [[nodes[0]['lon'], nodes[0]['lat']]]
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != 0:
                seq.append([nodes[node]['lon'], nodes[node]['lat']])
            idx = sol.Value(routing.NextVar(idx))
        seq.append([nodes[0]['lon'], nodes[0]['lat']])
        coords_per_route.append(seq)

    return coords_per_route, {"distancia_km": np.sum(dist_matrix)}

# --- INTERFAZ ---
st.title("🗺️ Optimización de Rutas Logísticas")
st.write("Carga un archivo Excel o CSV con la configuración de flota y pedidos para calcular la ruta óptima.")

with st.sidebar:
    st.header("📂 Cargar archivo")
    file = st.file_uploader("Sube tu archivo (Excel o CSV)", type=["xlsx", "csv"])

    if file is not None:
        df = pd.read_excel(file) if file.name.endswith(".xlsx") else pd.read_csv(file)
        st.success(f"{len(df)} registros cargados correctamente.")

        # --- Filtrar datos ---
        flota = df[df['tipo_registro'] == 'flota'].iloc[0]
        pedidos = df[df['tipo_registro'] == 'pedido']

        fleet_cfg = {
            "tipo": flota.get("tipo_vehiculo", "Camión"),
            "capacity_kg": flota.get("capacidad_kg", 1000),
            "capacity_m3": flota.get("capacidad_m3", 10),
            "velocidad": flota.get("velocidad_kmph", 40),
            "inicio_turno": flota.get("inicio_turno", "07:00"),
            "fin_turno": flota.get("fin_turno", "19:00"),
            "num_vehiculos": 3
        }

        st.session_state.puntos = pedidos.to_dict('records')

        st.info("Datos cargados: flota y pedidos listos para visualización.")

# --- MAPA ---
col1, col2 = st.columns((2,1))
with col1:
    st.subheader("🗺️ Mapa de pedidos")
    m = folium.Map(location=st.session_state.map_center, zoom_start=12)

    for p in st.session_state.puntos:
        folium.Marker(
            [p['lat'], p['lon']],
            popup=f"{p['nombre_pedido']} | {p['prioridad']} | {p['peso']}kg",
            tooltip=p['nombre_pedido']
        ).add_to(m)

    map_data = st_folium(m, height=550, width="100%")

with col2:
    st.subheader("🚀 Cálculo de Ruta")
    if st.button("Calcular Ruta (VRPTW)"):
        if not st.session_state.puntos:
            st.warning("Primero carga los pedidos.")
        else:
            centro = [st.session_state.puntos[0]['lat'], st.session_state.puntos[0]['lon']]
            rutas, metricas = solve_vrptw(centro, st.session_state.puntos, fleet_cfg)
            if rutas:
                for r in rutas:
                    folium.PolyLine([[lat, lon] for lon, lat in r], color="blue", weight=4).add_to(m)
                st.success("Rutas calculadas correctamente.")
                st.metric("Distancia estimada", f"{metricas['distancia_km']:.2f} km")
            else:
                st.error("No se pudo encontrar una ruta factible.")

st.subheader("📊 Pedidos cargados")
if 'puntos' in st.session_state and len(st.session_state.puntos) > 0:
    st.dataframe(pd.DataFrame(st.session_state.puntos))

