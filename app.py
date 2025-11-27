import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
# from openrouteservice import Client # No se usa en la versión Haversine
from math import radians, cos, sin, asin, sqrt
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Panel de Control de Rutas", page_icon="🚚", layout="wide")

# --- ESTADO INICIAL ---
if 'puntos' not in st.session_state: st.session_state.puntos = []
if 'map_center' not in st.session_state: st.session_state.map_center = [3.900, -76.300] # Buga aprox
if 'route_metrics' not in st.session_state: st.session_state.route_metrics = None # ¡CORRECCIÓN DE ATRIBUTO!

# --- FUNCIONES AUXILIARES ---
def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia Haversine en km."""
    # Convertir de grados a radianes
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
    
    # 1. Preparar parámetros de flota y nodos
    try:
        num_vehicles = int(fleet_cfg['Cantidad'])
        cap_kg = float(fleet_cfg['capacidad_kg'])
        speed_km_min = float(fleet_cfg['velocidad_kmh']) / 60.0
    except (TypeError, ValueError) as e:
        st.error(f"Error en la configuración de la flota: {e}. Asegura que 'Cantidad', 'capacidad_kg' y 'velocidad_kmh' sean números.")
        return None, None
    
    # Nodo 0: Depósito/Centro
    nodes = [{'lat': centro[0], 'lon': centro[1], 'demand': 0, 'service': 0, 
              'tw_start': fleet_cfg['turno_inicio'], 'tw_end': fleet_cfg['turno_fin']}]
    
    # Nodos 1..N: Pedidos
    for p in puntos:
        nodes.append({
            'lat': p['lat'], 'lon': p['lon'], 'demand': p['peso'], 'service': 15, # 15 min servicio fijo
            'tw_start': str(p.get('Tw_Start', '07:00')), 
            'tw_end': str(p.get('Tw_End', '19:00'))
        })

    N = len(nodes)
    
    # 2. Matrices de Distancia (km) y Tiempo (minutos)
    dist_matrix = np.zeros((N, N))
    time_matrix = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            if i != j:
                km = haversine_km(nodes[i]['lat'], nodes[i]['lon'], nodes[j]['lat'], nodes[j]['lon'])
                dist_matrix[i][j] = km
                # Tiempo = Distancia / Velocidad (en minutos)
                time_matrix[i][j] = (km / speed_km_min)
            # Agregar el tiempo de servicio al llegar al nodo destino (t)
            time_matrix[i][j] += nodes[j]['service']

    # 3. Configurar OR-Tools
    manager = pywrapcp.RoutingIndexManager(N, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_idx, to_idx):
        f, t = manager.IndexToNode(from_idx), manager.IndexToNode(to_idx)
        # La matriz ya incluye el tiempo de viaje + servicio en 't'
        return int(time_matrix[f][t])
        
    transit_idx = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)
    
    # Restricción de Tiempo (Dimension)
    # Penalidad alta si se excede el tiempo de la ventana (para evitar rutas imposibles)
    routing.AddDimension(transit_idx, 
                         int(time_str_to_minutes(fleet_cfg['turno_fin']) + 60), # Capacidad de tiempo max del vehículo
                         int(time_str_to_minutes(fleet_cfg['turno_fin']) + 60), # Límite de tiempo absoluto (para evitar overflow)
                         False, # No empieza en cero (el tiempo se acumula desde el inicio del turno)
                         "Time") 
    time_dim = routing.GetDimensionOrDie("Time")
    
    # Asignar Ventanas de Tiempo (Time Windows)
    for node_idx in range(N):
        idx = manager.NodeToIndex(node_idx)
        start = time_str_to_minutes(nodes[node_idx]['tw_start'])
        end = time_str_to_minutes(nodes[node_idx]['tw_end'])
        time_dim.CumulVar(idx).SetRange(start, end)
        
    # Restricción de Capacidad
    def demand_callback(from_idx):
        node = manager.IndexToNode(from_idx)
        return int(nodes[node]['demand'])
    
    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(demand_idx, 
                                            0, # Carga inicial (Depósito)
                                            [int(cap_kg)] * num_vehicles, # Capacidad de los vehículos
                                            True, # Sumar demanda a lo largo de la ruta
                                            "Capacity")

    # 4. Solución
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.time_limit.seconds = 5
    
    solution = routing.SolveWithParameters(search_parameters)

    if not solution:
        return None, None

    # 5. Extraer rutas y métricas
    rutas_coords = []
    distancia_total = 0
    
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route = []
        
        # Iniciar ruta en el Depósito
        route.append([nodes[0]['lat'], nodes[0]['lon']]) 
        
        while not routing.IsEnd(index):
            node_idx = manager.IndexToNode(index)
            # Agregar el punto de entrega actual (si no es el depósito)
            if node_idx != 0:
                route.append([nodes[node_idx]['lat'], nodes[node_idx]['lon']])
            
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            
            # Sumar distancia del segmento (aproximada para métrica)
            distancia_total += dist_matrix[manager.IndexToNode(previous_index)][manager.IndexToNode(index)]
            
        # Finalizar ruta volviendo al Depósito
        route.append([nodes[0]['lat'], nodes[0]['lon']])
        if len(route) > 2: # Solo si el vehículo salió
            rutas_coords.append(route)
        
    return rutas_coords, {"distancia_km": distancia_total}

# --- INTERFAZ ---
st.title("🗺️ Optimización Logística: Pedidos y Flota")

# Inicialización de variables para DataFrames
df_pedidos = None
df_flota = None
selected_fleet = None

with st.sidebar:
    st.header("📂 1. Cargar Datos")
    st.info("Sube un Excel con dos hojas: 'pedidos' y 'flota'")
    file = st.file_uploader("Archivo Excel (.xlsx)", type=["xlsx"])
    
    if file:
        try:
            # Leer ambas hojas (ajustado para leer Excel)
            xls = pd.ExcelFile(file)
            
            # Buscar hojas (flexible con mayúsculas/minúsculas y acentos)
            pedidos_sheet = next((s for s in xls.sheet_names if 'pedido' in s.lower()), None)
            flota_sheet = next((s for s in xls.sheet_names if 'flota' in s.lower()), None)

            if pedidos_sheet and flota_sheet:
                df_pedidos = pd.read_excel(file, sheet_name=pedidos_sheet)
                df_flota = pd.read_excel(file, sheet_name=flota_sheet)
                
                # --- AUTO-CORRECCIÓN DE COORDENADAS ---
                df_pedidos.columns = df_pedidos.columns.str.strip() # Limpiar encabezados
                
                # Intentar renombrar columnas de coordenadas antes de revisar
                df_pedidos = df_pedidos.rename(columns={
                    'Latitud': 'lat', 'Longitud': 'lon', 
                    'Peso (kg)': 'peso', 'Vol (m³)': 'vol',
                    'Tw_Start': 'Tw_Start', 'Tw_End': 'Tw_End'
                })
                
                # Si la latitud es > 15 (fuera de Colombia), dividimos por 10
                if 'lat' in df_pedidos.columns and df_pedidos['lat'].mean() > 15:
                    df_pedidos['lat'] = df_pedidos['lat'] / 10
                    df_pedidos['lon'] = df_pedidos['lon'] / 10
                    st.warning("⚠️ Detecté coordenadas mal escaladas (ej: 39.0 en vez de 3.9). Las corregí dividiendo por 10.")
                
                # Estandarizar columnas de flota
                df_flota.columns = df_flota.columns.str.strip().str.lower()
                
                st.success("✅ Datos cargados correctamente.")
                
                st.divider()
                st.header("🚚 2. Seleccionar Flota")
                
                # ----------------------------------------------------
                # SELECCIÓN DE FLOTA
                # ----------------------------------------------------
                
                if 'tipo_vehiculo' in df_flota.columns:
                    vehiculo_elegido = st.selectbox(
                        "¿Qué tipo de vehículo usarás?", 
                        df_flota['tipo_vehiculo'].unique()
                    )
                    
                    # Obtener la configuración de ese vehículo
                    selected_fleet_series = df_flota[df_flota['tipo_vehiculo'] == vehiculo_elegido].iloc[0]
                    selected_fleet = selected_fleet_series.to_dict()
                    
                    # Mostrar ficha técnica del vehículo seleccionado
                    st.caption(f"Configuración: **{int(selected_fleet.get('cantidad', 1))} {vehiculo_elegido}(s)**")
                    st.caption(f"Capacidad: {selected_fleet.get('capacidad_kg', 'N/A')} kg | Vel: {selected_fleet.get('velocidad_kmh', 'N/A')} km/h")
                    
                    # Guardar puntos en estado
                    st.session_state.puntos = df_pedidos.to_dict('records')
                    st.session_state.map_center = [df_pedidos.iloc[0]['lat'], df_pedidos.iloc[0]['lon']]
                else:
                    st.error("La hoja 'flota' debe contener la columna 'tipo_vehiculo'.")


            else:
                st.error("El Excel debe tener hojas llamadas 'pedidos' y 'flota'.")
        
        except Exception as e:
            st.error(f"Error procesando el archivo. Revisa el formato de tus datos: {e}")

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
        # Usamos el primer punto como depósito temporal para el ejemplo
        centro = [st.session_state.puntos[0]['lat'], st.session_state.puntos[0]['lon']] 
        
        rutas, metricas = solve_vrptw(centro, st.session_state.puntos, selected_fleet)
        
        if rutas:
            colors = ['red', 'green', 'blue', 'orange', 'purple', 'black']
            for i, ruta in enumerate(rutas):
                if len(ruta) > 1: # Solo dibujar rutas con al menos un punto
                    # Folium PolyLine espera [lat, lon]
                    folium.PolyLine(ruta, weight=5, color=colors[i % len(colors)], opacity=0.8,
                                    tooltip=f"Ruta {i+1}").add_to(m)
            
            st.session_state.route_metrics = metricas
            st.success("Rutas optimizadas con éxito. ¡Revisa el mapa!")
        else:
            st.session_state.route_metrics = None
            st.error("No se encontró solución factible (revisa capacidades, ventanas de tiempo, o si todos los puntos son accesibles).")

    st_folium(m, height=600, use_container_width=True)

with col2:
    st.subheader("📋 Resumen")
    
    # Esta línea ya no da error porque 'route_metrics' está inicializada a None
    if st.session_state.route_metrics: 
        st.metric("Distancia Total Estimada", f"{st.session_state.route_metrics['distancia_km']:.1f} km")
    
    if selected_fleet:
        st.info(f"Simulando con: **{selected_fleet.get('tipo_vehiculo', 'Vehículo Desconocido')}**")
    
    with st.expander("Ver Pedidos Cargados"):
        if df_pedidos is not None:
            st.dataframe(df_pedidos)
