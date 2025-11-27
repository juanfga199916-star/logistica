import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from openrouteservice import Client
from math import radians, cos, sin, asin, sqrt
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Panel de Control de Rutas", page_icon="🚚", layout="wide")

# --- ESTADO INICIAL ---
if 'puntos' not in st.session_state: st.session_state.puntos = []
if 'map_center' not in st.session_state: st.session_state.map_center = [3.900, -76.300] # Buga aprox

# --- FUNCIONES AUXILIARES ---
def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia Haversine en km."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 6371 * 2 * asin(sqrt(a))

def time_str_to_minutes(t):
    if isinstance(t, str):
        try:
            h, m = map(int, t.split(':'))
            return h*60 + m
        except: return 480 # Default 8:00
    return 480

def solve_vrptw(centro, puntos, fleet_cfg):
    """Resuelve VRPTW básico con OR-Tools."""
    # Convertir flota a parámetros
    num_vehicles = int(fleet_cfg['Cantidad'])
    cap_kg = float(fleet_cfg['capacidad_kg'])
    speed_km_min = float(fleet_cfg['velocidad_kmh']) / 60.0
    
    # 1. Crear Nodos (0 es el depósito)
    nodes = [{'lat': centro[0], 'lon': centro[1], 'demand': 0, 'service': 0, 
              'tw_start': fleet_cfg['turno_inicio'], 'tw_end': fleet_cfg['turno_fin']}]
    
    for p in puntos:
        nodes.append({
            'lat': p['lat'], 'lon': p['lon'], 'demand': p['peso'], 'service': 15, # 15 min servicio
            'tw_start': str(p.get('Tw_Start', '07:00')), 
            'tw_end': str(p.get('Tw_End', '19:00'))
        })

    N = len(nodes)
    
    # 2. Matrices de Distancia y Tiempo
    dist_matrix = np.zeros((N, N))
    time_matrix = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            if i != j:
                km = haversine_km(nodes[i]['lat'], nodes[i]['lon'], nodes[j]['lat'], nodes[j]['lon'])
                dist_matrix[i][j] = km
                # Tiempo = (Distancia / Velocidad) + Servicio
                travel_time = (km / speed_km_min)
                time_matrix[i][j] = travel_time

    # 3. Configurar OR-Tools
    manager = pywrapcp.RoutingIndexManager(N, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_idx, to_idx):
        f, t = manager.IndexToNode(from_idx), manager.IndexToNode(to_idx)
        return int(time_matrix[f][t] + nodes[f]['service']) # Service time added at node
        
    transit_idx = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)
    
    # Restricción de Tiempo (Dimension)
    routing.AddDimension(transit_idx, 10000, 10000, False, "Time") # Slack max grande para pruebas
    time_dim = routing.GetDimensionOrDie("Time")
    
    # Restricción de Capacidad
    def demand_callback(from_idx):
        node = manager.IndexToNode(from_idx)
        return int(nodes[node]['demand'])
    
    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, [int(cap_kg)]*num_vehicles, True, "Capacity")

    # Solución
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.time_limit.seconds = 5
    
    solution = routing.SolveWithParameters(search_parameters)

    if not solution:
        return None, None

    # Extraer rutas
    rutas_coords = []
    distancia_total = 0
    
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            node_idx = manager.IndexToNode(index)
            route.append([nodes[node_idx]['lat'], nodes[node_idx]['lon']])
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            # Sumar distancia (aproximada para métrica)
            distancia_total += dist_matrix[manager.IndexToNode(previous_index)][manager.IndexToNode(index)]
            
        # Añadir vuelta al depósito
        node_idx = manager.IndexToNode(index)
        route.append([nodes[node_idx]['lat'], nodes[node_idx]['lon']])
        rutas_coords.append(route)
        
    return rutas_coords, {"distancia_km": distancia_total}

# --- INTERFAZ ---
st.title("🗺️ Optimización Logística: Pedidos y Flota")

with st.sidebar:
    st.header("📂 1. Cargar Datos")
    st.info("Sube un Excel con dos hojas: 'pedidos' y 'flota'")
    file = st.file_uploader("Archivo Excel (.xlsx)", type=["xlsx"])
    
    df_pedidos = None
    df_flota = None
    selected_fleet = None

    if file:
        try:
            # Leer ambas hojas
            xls = pd.ExcelFile(file)
            sheet_names = [n.lower() for n in xls.sheet_names]
            
            # Buscar hojas (flexible con mayúsculas/minúsculas)
            pedidos_sheet = next((s for s in xls.sheet_names if 'pedido' in s.lower()), None)
            flota_sheet = next((s for s in xls.sheet_names if 'flota' in s.lower()), None)

            if pedidos_sheet and flota_sheet:
                df_pedidos = pd.read_excel(file, sheet_name=pedidos_sheet)
                df_flota = pd.read_excel(file, sheet_name=flota_sheet)
                
                # --- AUTO-CORRECCIÓN DE COORDENADAS ---
                # Si la latitud es > 15 (Colombia está entre -4 y 12), dividimos por 10
                if df_pedidos['Latitud'].mean() > 15:
                    df_pedidos['Latitud'] = df_pedidos['Latitud'] / 10
                    df_pedidos['Longitud'] = df_pedidos['Longitud'] / 10
                    st.warning("⚠️ Detecté coordenadas mal escaladas (ej: 39.0 en vez de 3.9). Las he corregido automáticamente.")

                # Renombrar columnas para estandarizar
                df_pedidos.columns = df_pedidos.columns.str.strip() # Quitar espacios
                df_pedidos = df_pedidos.rename(columns={
                    'Latitud': 'lat', 'Longitud': 'lon', 
                    'Peso (kg)': 'peso', 'Vol (m³)': 'vol'
                })
                
                st.success("✅ Datos cargados correctamente.")
                
                st.divider()
                st.header("🚚 2. Seleccionar Flota")
                # Permitir al usuario elegir qué vehículo usar de la tabla flota
                vehiculo_elegido = st.selectbox(
                    "¿Qué tipo de vehículo usarás?", 
                    df_flota['tipo_vehiculo'].unique()
                )
                
                # Obtener la configuración de ese vehículo
                selected_fleet = df_flota[df_flota['tipo_vehiculo'] == vehiculo_elegido].iloc[0].to_dict()
                
                # Mostrar ficha técnica del vehículo seleccionado
                st.caption(f"Configuración: {int(selected_fleet['Cantidad'])} {vehiculo_elegido}(s)")
                st.caption(f"Capacidad: {selected_fleet['capacidad_kg']} kg | Vel: {selected_fleet['velocidad_kmh']} km/h")
                
                # Guardar en sesión
                st.session_state.puntos = df_pedidos.to_dict('records')
                st.session_state.map_center = [df_pedidos.iloc[0]['lat'], df_pedidos.iloc[0]['lon']]

            else:
                st.error("El Excel debe tener hojas llamadas 'pedidos' y 'flota'.")
        
        except Exception as e:
            st.error(f"Error procesando el archivo: {e}")

# --- VISUALIZACIÓN ---
col1, col2 = st.columns((3, 1))

with col1:
    m = folium.Map(location=st.session_state.map_center, zoom_start=11)
    
    # Dibujar pedidos
    for p in st.session_state.puntos:
        folium.CircleMarker(
            location=[p['lat'], p['lon']],
            radius=5,
            color="blue",
            fill=True,
            tooltip=f"{p.get('Nombre Pedido', 'Pedido')} | {p['peso']}kg"
        ).add_to(m)

    # Lógica de cálculo
    if selected_fleet and st.button("🚀 Calcular Rutas"):
        centro = [st.session_state.puntos[0]['lat'], st.session_state.puntos[0]['lon']]
        rutas, metricas = solve_vrptw(centro, st.session_state.puntos, selected_fleet)
        
        if rutas:
            colors = ['red', 'green', 'blue', 'orange', 'purple']
            for i, ruta in enumerate(rutas):
                if len(ruta) > 2: # Solo dibujar si sale del depósito
                    folium.PolyLine(ruta, weight=5, color=colors[i % len(colors)], opacity=0.8).add_to(m)
            st.session_state.route_metrics = metricas
            st.success("Rutas optimizadas con éxito")
        else:
            st.error("No se encontró solución factible (revisa capacidades o ventanas de tiempo).")

    st_folium(m, height=600, use_container_width=True)

with col2:
    st.subheader("📋 Resumen")
    if st.session_state.route_metrics:
        st.metric("Distancia Total Estimada", f"{st.session_state.route_metrics['distancia_km']:.1f} km")
    
    if selected_fleet:
        st.info(f"Simulando con: **{selected_fleet['tipo_vehiculo']}**")
    
    with st.expander("Ver Datos de Pedidos"):
        if df_pedidos is not None:
            st.dataframe(df_pedidos[['Nombre Pedido', 'peso', 'lat', 'lon']])
