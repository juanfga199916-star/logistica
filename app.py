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
if 'map_center' not in st.session_state: st.session_state.map_center = [3.900, -76.300]
if 'route_metrics' not in st.session_state: st.session_state.route_metrics = None
if 'cedis' not in st.session_state: st.session_state.cedis = {'lat': 3.900, 'lon': -76.300, 'nombre': 'CEDIS Inicial (Buga)'}
if 'df_pedidos' not in st.session_state: st.session_state.df_pedidos = None # Nuevo estado para DF
if 'df_flota' not in st.session_state: st.session_state.df_flota = None     # Nuevo estado para DF

# Colores fijos para las rutas
ROUTE_COLORS = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33', '#A65628', '#F781BF', '#999999', '#66C2A5']

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
        except: return 420
    return 420

def solve_vrptw(centro, puntos, df_flota):
    """Resuelve VRPTW usando toda la flota disponible."""
    
    # 1. Crear Pool de Vehículos (homogéneos) y mapeo
    vehicle_pool = []
    vehicle_types = []
    vehicle_capacities = []
    vehicle_speeds = []
    
    # Asumimos que todos los vehículos tienen el mismo turno
    if df_flota.empty:
        st.error("La flota está vacía.")
        return None, None
    
    # Parámetros del depósito basados en el primer tipo de vehículo (asumimos uniformidad en turno)
    depot_start_time = df_flota.iloc[0]['turno_inicio']
    depot_end_time = df_flota.iloc[0]['turno_fin']
    
    # Construir el pool
    for idx, row in df_flota.iterrows():
        for i in range(int(row['cantidad'])):
            vehicle_pool.append({
                'type': row['tipo_vehiculo'],
                'capacity_kg': float(row['capacidad_kg']),
                'speed_kmph': float(row['velocidad_kmh']),
                'start_time': row['turno_inicio'],
                'end_time': row['turno_fin']
            })
            vehicle_capacities.append(int(row['capacidad_kg']))
            vehicle_speeds.append(float(row['velocidad_kmh']))
            vehicle_types.append(row['tipo_vehiculo'])
    
    num_vehicles = len(vehicle_pool)
    if num_vehicles == 0:
        st.error("No hay vehículos disponibles en la flota.")
        return None, None

    # 2. Crear Nodos
    # Nodo 0: Depósito/Centro
    nodes = [{'lat': centro['lat'], 'lon': centro['lon'], 'demand': 0, 'service': 0, 
              'tw_start': depot_start_time, 'tw_end': depot_end_time}]
    
    # Nodos 1..N: Pedidos
    for p in puntos:
        nodes.append({
            'lat': p['lat'], 'lon': p['lon'], 'demand': p['peso'], 'service': 15,
            'tw_start': str(p.get('Tw_Start', '07:00')), 
            'tw_end': str(p.get('Tw_End', '19:00'))
        })
    N = len(nodes)
    
    # 3. Matrices de Distancia y Tiempo (usando la velocidad del vehículo específico)
    # Debemos calcular la matriz de tiempo por cada vehículo si las velocidades son diferentes, 
    # pero OR-Tools solo permite una matriz de tránsito. Para simplificar, usamos la velocidad promedio
    # o la del vehículo más lento (conservador). Aquí usamos una velocidad fija (la del primer vehículo).
    
    # Usamos la velocidad del primer vehículo del pool para la matriz de costos.
    # Nota: Para flotas heterogéneas reales, se debe usar la dimensión de Costo (Distance/Time) por vehículo.
    avg_speed_km_min = vehicle_pool[0]['speed_kmph'] / 60.0
    
    dist_matrix = np.zeros((N, N))
    time_matrix = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            if i != j:
                km = haversine_km(nodes[i]['lat'], nodes[i]['lon'], nodes[j]['lat'], nodes[j]['lon'])
                dist_matrix[i][j] = km
                time_matrix[i][j] = (km / avg_speed_km_min)
            time_matrix[i][j] += nodes[j]['service']

    # 4. Configurar OR-Tools
    manager = pywrapcp.RoutingIndexManager(N, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_idx, to_idx):
        f, t = manager.IndexToNode(from_idx), manager.IndexToNode(to_idx)
        return int(time_matrix[f][t])
        
    transit_idx = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)
    
    # Dimensión de Tiempo
    max_time = int(time_str_to_minutes(depot_end_time) + 60)
    routing.AddDimension(transit_idx, max_time, max_time, True, "Time") # True para permitir 'slack'
    time_dim = routing.GetDimensionOrDie("Time")
    
    for node_idx in range(N):
        idx = manager.NodeToIndex(node_idx)
        start = time_str_to_minutes(nodes[node_idx]['tw_start'])
        end = time_str_to_minutes(nodes[node_idx]['tw_end'])
        time_dim.CumulVar(idx).SetRange(start, end)
        
    # Dimensión de Capacidad
    def demand_callback(from_idx):
        node = manager.IndexToNode(from_idx)
        return int(nodes[node]['demand'])
    
    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    
    # Capacidad por vehículo
    routing.AddDimensionWithVehicleCapacity(demand_idx, 
                                            0, # Carga inicial
                                            vehicle_capacities, # Lista de capacidades de cada vehículo
                                            True, 
                                            "Capacity")
    
    # 5. Penalizar el no uso de un nodo (para forzar la visita)
    # Solo penalizamos si el nodo NO es el depósito
    for node_idx in range(1, N): 
        routing.AddDisjunction([manager.NodeToIndex(node_idx)], 1000000) # Penalidad alta

    # 6. Solución
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.time_limit.seconds = 5
    
    solution = routing.SolveWithParameters(search_parameters)

    if not solution:
        return None, None

    # 7. Extraer rutas, métricas y el vehículo utilizado
    rutas_info = []
    distancia_total = 0
    vehiculos_usados = []
    
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route_coords = []
        route_served = False
        
        while not routing.IsEnd(index):
            node_idx = manager.IndexToNode(index)
            
            # Si el vehículo se mueve, marca la ruta como servida
            if node_idx != 0:
                route_coords.append([nodes[node_idx]['lat'], nodes[node_idx]['lon']])
                route_served = True

            previous_index = index
            index = solution.Value(routing.NextVar(index))
            
            # Sumar distancia solo si hay movimiento
            if previous_index != index:
                 distancia_total += dist_matrix[manager.IndexToNode(previous_index)][manager.IndexToNode(index)]
        
        # Guardar ruta si fue utilizada
        if route_served:
            # Añadir CEDIS al inicio y final
            route_coords.insert(0, [centro['lat'], centro['lon']])
            route_coords.append([centro['lat'], centro['lon']])
            
            vehiculos_usados.append(vehicle_pool[vehicle_id]['type'])
            rutas_info.append({
                'coords': route_coords,
                'vehicle_type': vehicle_pool[vehicle_id]['type'],
                'vehicle_id': vehicle_id + 1
            })
            
    return rutas_info, {"distancia_km": distancia_total, "vehiculos_usados": vehiculos_usados}

# --- INTERFAZ ---
col1, col2 = st.columns((3, 1))

with st.sidebar:
    st.header("📂 1. Cargar Datos")
    st.info("Sube un Excel con hojas: 'pedidos' y 'flota'")
    file = st.file_uploader("Archivo Excel (.xlsx)", type=["xlsx"])
    
    # Lógica de carga de datos
    if file:
        try:
            xls = pd.ExcelFile(file)
            pedidos_sheet = next((s for s in xls.sheet_names if 'pedido' in s.lower()), None)
            flota_sheet = next((s for s in xls.sheet_names if 'flota' in s.lower()), None)

            if pedidos_sheet and flota_sheet:
                df_pedidos_loaded = pd.read_excel(file, sheet_name=pedidos_sheet)
                df_flota_loaded = pd.read_excel(file, sheet_name=flota_sheet)
                
                # Preprocesamiento
                df_pedidos_loaded.columns = df_pedidos_loaded.columns.str.strip() 
                df_pedidos_loaded = df_pedidos_loaded.rename(columns={
                    'Latitud': 'lat', 'Longitud': 'lon', 
                    'Peso (kg)': 'peso', 'Vol (m³)': 'vol',
                })
                if 'lat' in df_pedidos_loaded.columns and df_pedidos_loaded['lat'].mean() > 15:
                    df_pedidos_loaded['lat'] = df_pedidos_loaded['lat'] / 10
                    df_pedidos_loaded['lon'] = df_pedidos_loaded['lon'] / 10
                    st.warning("⚠️ Coordenadas corregidas.")
                
                df_flota_loaded.columns = df_flota_loaded.columns.str.strip().str.lower()
                
                # Guardar en estado de sesión (SOLUCIÓN AL ERROR DE ATRIBUTO)
                st.session_state.df_pedidos = df_pedidos_loaded
                st.session_state.df_flota = df_flota_loaded
                st.session_state.puntos = st.session_state.df_pedidos.to_dict('records')
                
                st.success("✅ Datos cargados. La flota será asignada automáticamente.")

            else:
                st.error("El Excel debe tener hojas llamadas 'pedidos' y 'flota'.")
        
        except Exception as e:
            st.error(f"Error procesando el archivo. Revisa el formato de tus datos: {e}")
            
    # --- 2. Ingreso Manual del CEDIS ---
    st.divider()
    st.header("📍 2. Ubicación del CEDIS")
    col_lat, col_lon = st.columns(2)
    
    cedis_lat = col_lat.number_input("Latitud CEDIS", value=st.session_state.cedis['lat'], format="%.4f")
    cedis_lon = col_lon.number_input("Longitud CEDIS", value=st.session_state.cedis['lon'], format="%.4f")
    st.session_state.cedis = {'lat': cedis_lat, 'lon': cedis_lon, 'nombre': 'CEDIS Personalizado'}
    
    # --- 3. Costos y Cálculo ---
    st.divider()
    st.header("💰 3. Costo y Cálculo")
    
    costo_por_km = st.slider(
        "Costo Operacional por Kilómetro ($/km)", 
        min_value=500.0, max_value=5000.0, value=2500.0, step=100.0, format="$ %.0f"
    )

# --- VISTA PRINCIPAL ---
with col1:
    st.subheader("🗺️ Visualización de Rutas")
    
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
    if st.session_state.route_metrics and st.session_state.route_metrics.get('rutas_info'):
        for i, ruta_info in enumerate(st.session_state.route_metrics['rutas_info']):
            folium.PolyLine(
                ruta_info['coords'], 
                weight=5, 
                color=ROUTE_COLORS[i % len(ROUTE_COLORS)], 
                opacity=0.8,
                tooltip=f"Ruta {ruta_info['vehicle_id']} ({ruta_info['vehicle_type']})"
            ).add_to(m)

    st_folium(m, height=600, use_container_width=True)

with col2:
    st.subheader("🚀 Cálculo y Resultados")
    
    # --- BOTÓN DE CÁLCULO DEDICADO ---
    if st.button("CALCULAR RUTA ÓPTIMA", type="primary"):
        if not st.session_state.puntos or st.session_state.df_flota is None:
            st.error("Por favor, carga los pedidos y la flota.")
        else:
            with st.spinner("Calculando asignación de flota y rutas..."):
                rutas_info, metricas = solve_vrptw(st.session_state.cedis, st.session_state.puntos, st.session_state.df_flota)
            
            if rutas_info:
                st.session_state.route_metrics = {
                    'distancia_km': metricas['distancia_km'],
                    'costo_por_km': costo_por_km,
                    'rutas_info': rutas_info,
                    'vehiculos_usados': metricas['vehiculos_usados']
                }
                st.success(f"Cálculo finalizado. {len(rutas_info)} vehículos utilizados.")
            else:
                st.session_state.route_metrics = None
                st.error("No se encontró solución factible. Revisa si la flota es suficiente o las ventanas de tiempo son muy restrictivas.")
    # ------------------------------------

    st.divider()
    
    # --- RESULTADOS ---
    if st.session_state.route_metrics: 
        distancia = st.session_state.route_metrics['distancia_km']
        costo_total = distancia * st.session_state.route_metrics['costo_por_km']
        num_rutas = len(st.session_state.route_metrics['rutas_info'])
        
        st.metric("Distancia Total Estimada", f"{distancia:.1f} km", delta=f"{num_rutas} Rutas")
        st.metric("Costo Operacional Total", f"$ {costo_total:,.0f} COP")
        
        st.subheader("Flota Asignada")
        
        # Agrupar y contar los vehículos utilizados
        df_usados = pd.Series(st.session_state.route_metrics['vehiculos_usados']).value_counts().reset_index()
        df_usados.columns = ['Tipo de Vehículo', 'Cantidad Utilizada']
        st.dataframe(df_usados, hide_index=True)

    with st.expander("Ver Pedidos Cargados"):
        if st.session_state.df_pedidos is not None:
            st.dataframe(st.session_state.df_pedidos[['Nombre Pedido', 'peso', 'lat', 'lon', 'Tw_Start', 'Tw_End']])
