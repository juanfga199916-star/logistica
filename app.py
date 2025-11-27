import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from math import radians, cos, sin, asin, sqrt
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Panel de Control de Rutas", page_icon="🚚", layout="wide")

# --- ESTADO INICIAL ---
# Inicialización robusta de todas las variables de estado
if 'puntos' not in st.session_state: st.session_state.puntos = []
if 'map_center' not in st.session_state: st.session_state.map_center = [3.900, -76.300]
if 'route_metrics' not in st.session_state: st.session_state.route_metrics = None
if 'cedis' not in st.session_state: st.session_state.cedis = {'lat': 3.900, 'lon': -76.300, 'nombre': 'CEDIS Inicial (Buga)'}
if 'df_pedidos' not in st.session_state: st.session_state.df_pedidos = None
if 'df_flota' not in st.session_state: st.session_state.df_flota = None

# Colores predefinidos para las rutas
ROUTE_COLORS = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33', '#A65628', '#F781BF', '#999999', '#66C2A5']

# --- FUNCIONES AUXILIARES ---
def haversine_km(lat1, lon1, lat2, lon2):
    """Calcula la distancia Haversine (geodésica/línea recta) en km."""
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
        except: return 420 # 7:00 AM
    return 420

def solve_vrptw(centro, puntos, df_flota):
    """
    Resuelve el VRPTW utilizando la flota completa disponible. 
    OR-Tools asignará las rutas y los vehículos necesarios.
    """
    
    # 1. Crear Pool de Vehículos (Lista de todos los vehículos individuales)
    vehicle_pool = []
    vehicle_capacities = []
    vehicle_speeds = []
    
    if df_flota.empty:
        st.error("La flota está vacía. No se puede calcular la ruta.")
        return None, None
    
    depot_start_time = df_flota.iloc[0]['turno_inicio']
    depot_end_time = df_flota.iloc[0]['turno_fin']
    
    # Construir el pool de vehículos individuales
    for idx, row in df_flota.iterrows():
        for i in range(int(row['cantidad'])):
            vehicle_pool.append({
                'type': row['tipo_vehiculo'],
                'capacity_kg': float(row['capacidad_kg']),
                'speed_kmph': float(row['velocidad_kmh']),
            })
            vehicle_capacities.append(int(row['capacidad_kg']))
            vehicle_speeds.append(float(row['velocidad_kmh']))
    
    num_vehicles = len(vehicle_pool)
    if num_vehicles == 0:
        st.error("No hay vehículos disponibles en la flota.")
        return None, None
    
    # Usamos la velocidad promedio de la flota para calcular la matriz de tiempo
    # (Simplificación para flotas heterogéneas, lo ideal es usar costos por vehículo)
    avg_speed_km_min = np.mean(vehicle_speeds) / 60.0
    
    # 2. Crear Nodos
    nodes = [{'lat': centro['lat'], 'lon': centro['lon'], 'demand': 0, 'service': 0, 
              'tw_start': depot_start_time, 'tw_end': depot_end_time}]
    
    for p in puntos:
        nodes.append({
            'lat': p['lat'], 'lon': p['lon'], 'demand': p['peso'], 'service': 15, # Tiempo de servicio fijo
            'tw_start': str(p.get('Tw_Start', '07:00')), 
            'tw_end': str(p.get('Tw_End', '19:00'))
        })
    N = len(nodes)
    
    # 3. Matrices de Distancia y Tiempo
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
    routing.AddDimension(transit_idx, max_time, max_time, True, "Time")
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
    
    routing.AddDimensionWithVehicleCapacity(demand_idx, 
                                            0, 
                                            vehicle_capacities,
                                            True, 
                                            "Capacity")
    
    # Penalidad alta por no entregar (permite a OR-Tools dejar pedidos si no son factibles)
    for node_idx in range(1, N): 
        routing.AddDisjunction([manager.NodeToIndex(node_idx)], 1000000)

    # 5. Solución
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.time_limit.seconds = 5
    
    solution = routing.SolveWithParameters(search_parameters)

    if not solution:
        return None, None

    # 6. Extraer rutas, métricas y el vehículo utilizado
    rutas_info = []
    distancia_total = 0
    vehiculos_usados = []
    
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route_coords = []
        route_served = False
        
        while not routing.IsEnd(index):
            node_idx = manager.IndexToNode(index)
            
            if node_idx != 0:
                route_coords.append([nodes[node_idx]['lat'], nodes[node_idx]['lon']])
                route_served = True

            previous_index = index
            index = solution.Value(routing.NextVar(index))
            
            if previous_index != index:
                 distancia_total += dist_matrix[manager.IndexToNode(previous_index)][manager.IndexToNode(index)]
        
        # Guardar ruta si fue utilizada
        if route_served:
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
st.title("🗺️ Optimización Logística: Pedidos y Flota")

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
                
                # --- Preprocesamiento de PEDIDOS (Corrección de KeyError) ---
                df_pedidos_loaded.columns = df_pedidos_loaded.columns.str.strip() 
                
                # Renombrar columnas clave a minúsculas y estandarizar
                df_pedidos_loaded = df_pedidos_loaded.rename(columns={
                    'Latitud': 'lat', 
                    'Longitud': 'lon', 
                    'Peso (kg)': 'peso', 
                    'Vol (m³)': 'vol',
                    'Nombre Pedido': 'nombre_pedido' 
                })
                
                # Fallback para columna 'nombre_pedido' si no existe
                if 'nombre_pedido' not in df_pedidos_loaded.columns:
                    try:
                        df_pedidos_loaded = df_pedidos_loaded.rename(columns={col: 'nombre_pedido' for col in df_pedidos_loaded.columns if 'nombre' in col.lower()})
                    except:
                        df_pedidos_loaded['nombre_pedido'] = 'Pedido ' + df_pedidos_loaded.index.astype(str)

                # Corrección de escala de coordenadas
                if 'lat' in df_pedidos_loaded.columns and df_pedidos_loaded['lat'].mean() > 15:
                    df_pedidos_loaded['lat'] = df_pedidos_loaded['lat'] / 10
                    df_pedidos_loaded['lon'] = df_pedidos_loaded['lon'] / 10
                    st.warning("⚠️ Coordenadas corregidas (división por 10).")
                
                # --- Preprocesamiento de FLOTA (Corrección de ValueError) ---
                df_flota_loaded.columns = df_flota_loaded.columns.str.strip().str.lower()
                
                # Limpiar y convertir a tipos numéricos, rellenando NaN con valores seguros
                df_flota_loaded['cantidad'] = df_flota_loaded['cantidad'].fillna(0).astype(int)
                df_flota_loaded['capacidad_kg'] = df_flota_loaded['capacidad_kg'].fillna(0).astype(float)
                df_flota_loaded['velocidad_kmh'] = df_flota_loaded['velocidad_kmh'].fillna(40).astype(float)
                
                # Guardar en estado de sesión
                st.session_state.df_pedidos = df_pedidos_loaded
                st.session_state.df_flota = df_flota_loaded
                st.session_state.puntos = st.session_state.df_pedidos.to_dict('records')
                
                st.success("✅ Datos cargados. La flota será asignada automáticamente.")

            else:
                st.error("El Excel debe tener hojas llamadas 'pedidos' y 'flota'.")
        
        except Exception as e:
            st.error(f"Error procesando el archivo: {e}")
            
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
            tooltip=f"{p.get('nombre_pedido', 'Pedido')} | {p['peso']}kg"
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
            # Se usa la columna 'nombre_pedido' estandarizada
            st.dataframe(st.session_state.df_pedidos[['nombre_pedido', 'peso', 'lat', 'lon', 'Tw_Start', 'Tw_End']])
