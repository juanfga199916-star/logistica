import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from math import radians, cos, sin, asin, sqrt
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Panel de Control de Rutas", page_icon="🚚", layout="wide")

# --- ESTADO INICIAL ---
if 'puntos' not in st.session_state: st.session_state.puntos = []
if 'map_center' not in st.session_state: st.session_state.map_center = [3.900, -76.300] # Buga aprox
if 'route_metrics' not in st.session_state: st.session_state.route_metrics = None
if 'cedis' not in st.session_state: st.session_state.cedis = {'lat': 3.900, 'lon': -76.300, 'nombre': 'CEDIS Inicial (Buga)'} # Depósito inicial

# --- FUNCIONES AUXILIARES ---
def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia Haversine en km."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 6371 * 2 * asin(sqrt(a))

def time_str_to_minutes(t):
    """Convierte un string de tiempo (HH:MM) a minutos desde la medianoche."""
    if isinstance(t, str):
        try:
            h, m = map(int, t.split(':'))
            return h*60 + m
        except: return 420 # Default 7:00 (420 min)
    return 420

def solve_vrptw(centro, puntos, fleet_cfg):
    """Resuelve VRPTW básico con OR-Tools (usando distancias Haversine)."""
    
    try:
        num_vehicles = int(fleet_cfg['Cantidad'])
        cap_kg = float(fleet_cfg['capacidad_kg'])
        speed_km_min = float(fleet_cfg['velocidad_kmh']) / 60.0
    except (TypeError, ValueError) as e:
        st.error(f"Error en la configuración de la flota: {e}. Asegura que 'Cantidad', 'capacidad_kg' y 'velocidad_kmh' sean números.")
        return None, None
    
    # Nodo 0: Depósito/Centro
    nodes = [{'lat': centro['lat'], 'lon': centro['lon'], 'demand': 0, 'service': 0, 
              'tw_start': fleet_cfg['turno_inicio'], 'tw_end': fleet_cfg['turno_fin']}]
    
    # Nodos 1..N: Pedidos
    for p in puntos:
        nodes.append({
            'lat': p['lat'], 'lon': p['lon'], 'demand': p['peso'], 'service': 15, # 15 min servicio fijo
            'tw_start': str(p.get('Tw_Start', '07:00')), 
            'tw_end': str(p.get('Tw_End', '19:00'))
        })

    N = len(nodes)
    
    # Matrices de Distancia (km) y Tiempo (minutos)
    dist_matrix = np.zeros((N, N))
    time_matrix = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            if i != j:
                km = haversine_km(nodes[i]['lat'], nodes[i]['lon'], nodes[j]['lat'], nodes[j]['lon'])
                dist_matrix[i][j] = km
                time_matrix[i][j] = (km / speed_km_min)
            time_matrix[i][j] += nodes[j]['service'] # Agregar el tiempo de servicio

    # Configurar OR-Tools
    manager = pywrapcp.RoutingIndexManager(N, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_idx, to_idx):
        f, t = manager.IndexToNode(from_idx), manager.IndexToNode(to_idx)
        return int(time_matrix[f][t])
        
    transit_idx = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)
    
    # Dimensiones (Tiempo y Capacidad)
    max_time = int(time_str_to_minutes(fleet_cfg['turno_fin']) + 60)
    routing.AddDimension(transit_idx, max_time, max_time, False, "Time") 
    time_dim = routing.GetDimensionOrDie("Time")
    
    for node_idx in range(N):
        idx = manager.NodeToIndex(node_idx)
        start = time_str_to_minutes(nodes[node_idx]['tw_start'])
        end = time_str_to_minutes(nodes[node_idx]['tw_end'])
        time_dim.CumulVar(idx).SetRange(start, end)
        
    def demand_callback(from_idx):
        node = manager.IndexToNode(from_idx)
        return int(nodes[node]['demand'])
    
    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, [int(cap_kg)] * num_vehicles, True, "Capacity")

    # Solución
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.time_limit.seconds = 5
    
    solution = routing.SolveWithParameters(search_parameters)

    if not solution:
        return None, None

    # Extraer rutas y métricas
    rutas_coords = []
    distancia_total = 0
    
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route = []
        route.append([centro['lat'], centro['lon']]) # Iniciar en CEDIS
        
        while not routing.IsEnd(index):
            node_idx = manager.IndexToNode(index)
            if node_idx != 0:
                route.append([nodes[node_idx]['lat'], nodes[node_idx]['lon']])
            
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            
            distancia_total += dist_matrix[manager.IndexToNode(previous_index)][manager.IndexToNode(index)]
            
        route.append([centro['lat'], centro['lon']]) # Finalizar en CEDIS
        if len(route) > 2: 
            rutas_coords.append(route)
        
    return rutas_coords, {"distancia_km": distancia_total}

# --- INTERFAZ ---
st.title("🗺️ Optimización Logística: Pedidos y Flota")

# Inicialización de variables para DataFrames
df_pedidos = None
df_flota = None
selected_fleet = None

# --- SIDEBAR: CONFIGURACIÓN Y CÁLCULO ---
with st.sidebar:
    st.header("📂 1. Cargar Datos")
    st.info("Sube un Excel con hojas: 'pedidos' y 'flota'")
    file = st.file_uploader("Archivo Excel (.xlsx)", type=["xlsx"])
    
    if file:
        try:
            xls = pd.ExcelFile(file)
            pedidos_sheet = next((s for s in xls.sheet_names if 'pedido' in s.lower()), None)
            flota_sheet = next((s for s in xls.sheet_names if 'flota' in s.lower()), None)

            if pedidos_sheet and flota_sheet:
                df_pedidos = pd.read_excel(file, sheet_name=pedidos_sheet)
                df_flota = pd.read_excel(file, sheet_name=flota_sheet)
                
                # --- Preprocesamiento (Coordenadas y Nombres) ---
                df_pedidos.columns = df_pedidos.columns.str.strip() 
                df_pedidos = df_pedidos.rename(columns={
                    'Latitud': 'lat', 'Longitud': 'lon', 
                    'Peso (kg)': 'peso', 'Vol (m³)': 'vol',
                })
                if 'lat' in df_pedidos.columns and df_pedidos['lat'].mean() > 15:
                    df_pedidos['lat'] = df_pedidos['lat'] / 10
                    df_pedidos['lon'] = df_pedidos['lon'] / 10
                    st.warning("⚠️ Coordenadas corregidas.")
                df_flota.columns = df_flota.columns.str.strip().str.lower()
                st.success("✅ Datos cargados.")
                
                # --- 2. Seleccionar Flota ---
                st.divider()
                st.header("🚚 2. Configuración de Flota")
                if 'tipo_vehiculo' in df_flota.columns:
                    vehiculo_elegido = st.selectbox(
                        "Tipo de vehículo para la simulación:", 
                        df_flota['tipo_vehiculo'].unique()
                    )
                    selected_fleet_series = df_flota[df_flota['tipo_vehiculo'] == vehiculo_elegido].iloc[0]
                    selected_fleet = selected_fleet_series.to_dict()
                    st.caption(f"Capacidad: {selected_fleet.get('capacidad_kg')} kg | Vehículos: {selected_fleet.get('cantidad')}")
                    
                    st.session_state.puntos = df_pedidos.to_dict('records')
                else:
                    st.error("La hoja 'flota' debe contener la columna 'tipo_vehiculo'.")

            else:
                st.error("El Excel debe tener hojas llamadas 'pedidos' y 'flota'.")
        
        except Exception as e:
            st.error(f"Error procesando el archivo: {e}")
            
    # --- 3. Ingreso Manual del CEDIS ---
    st.divider()
    st.header("📍 3. Ubicación del CEDIS")
    col_lat, col_lon = st.columns(2)
    
    default_lat = st.session_state.cedis['lat']
    default_lon = st.session_state.cedis['lon']

    cedis_lat = col_lat.number_input("Latitud CEDIS", value=default_lat, format="%.4f")
    cedis_lon = col_lon.number_input("Longitud CEDIS", value=default_lon, format="%.4f")
    
    st.session_state.cedis = {'lat': cedis_lat, 'lon': cedis_lon, 'nombre': 'CEDIS Personalizado'}
    
    # --- 4. Costos y Cálculo ---
    st.divider()
    st.header("💰 4. Costo y Cálculo")
    
    costo_por_km = st.slider(
        "Costo Operacional por Kilómetro ($/km)", 
        min_value=500.0, max_value=5000.0, value=2500.0, step=100.0, format="$ %.0f"
    )

# --- VISTA PRINCIPAL ---
col1, col2 = st.columns((3, 1))

with col1:
    st.subheader("🗺️ Visualización de Rutas")
    
    # Centrar el mapa en el CEDIS actual
    m = folium.Map(location=[st.session_state.cedis['lat'], st.session_state.cedis['lon']], zoom_start=11)
    
    # Dibujar CEDIS
    folium.Marker(
        location=[st.session_state.cedis['lat'], st.session_state.cedis['lon']],
        popup=st.session_state.cedis['nombre'],
        icon=folium.Icon(color='green', icon='warehouse', prefix='fa')
    ).add_to(m)
    
    # Dibujar pedidos
    for p in st.session_state.puntos:
        folium.CircleMarker(
            location=[p['lat'], p['lon']],
            radius=5,
            color="blue",
            fill=True,
            tooltip=f"{p.get('Nombre Pedido', 'Pedido')} | {p['peso']}kg"
        ).add_to(m)

    # Dibuja rutas si están calculadas
    if st.session_state.route_metrics and st.session_state.route_metrics.get('rutas_coords'):
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'black']
        for i, ruta in enumerate(st.session_state.route_metrics['rutas_coords']):
            if len(ruta) > 1:
                # Usar PolyLine para graficar las líneas rectas (Haversine)
                folium.PolyLine(
                    ruta, 
                    weight=5, 
                    color=colors[i % len(colors)], 
                    opacity=0.8,
                    tooltip=f"Ruta {i+1}"
                ).add_to(m)

    st_folium(m, height=600, use_container_width=True)

with col2:
    st.subheader("🚀 Cálculo y Resultados")
    
    # --- BOTÓN DE CÁLCULO DEDICADO ---
    if st.button("CALCULAR RUTA ÓPTIMA", type="primary"):
        if not st.session_state.puntos or not selected_fleet:
            st.error("Por favor, carga los pedidos y selecciona la flota primero.")
        else:
            with st.spinner("Calculando rutas, capacidad y ventanas de tiempo con OR-Tools..."):
                rutas, metricas = solve_vrptw(st.session_state.cedis, st.session_state.puntos, selected_fleet)
            
            if rutas:
                st.session_state.route_metrics = {
                    'distancia_km': metricas['distancia_km'],
                    'costo_por_km': costo_por_km,
                    'rutas_coords': rutas # Guardar las coordenadas para el dibujo
                }
                st.success("Cálculo finalizado con éxito.")
            else:
                st.session_state.route_metrics = None
                st.error("No se encontró solución factible. Revisa si la capacidad o las ventanas de tiempo son muy restrictivas.")
    # ------------------------------------

    st.divider()
    
    # --- RESULTADOS ---
    if st.session_state.route_metrics: 
        distancia = st.session_state.route_metrics['distancia_km']
        costo_total = distancia * st.session_state.route_metrics['costo_por_km']
        num_rutas = len(st.session_state.route_metrics['rutas_coords'])
        
        st.metric("Distancia Total Estimada", f"{distancia:.1f} km", delta=f"{num_rutas} Rutas")
        st.metric("Costo Operacional Total", f"$ {costo_total:,.0f} COP")
    
    if selected_fleet:
        st.info(f"Simulando con: **{selected_fleet.get('tipo_vehiculo', 'Vehículo Desconocido')}**")
    
    with st.expander("Ver Pedidos Cargados"):
        if df_pedidos is not None:
            st.dataframe(df_pedidos[['Nombre Pedido', 'peso', 'lat', 'lon', 'Tw_Start', 'Tw_End']])
