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
from typing import Tuple, List, Dict, Any, Optional

# --- CONFIGURACIÓN GLOBAL ---
st.set_page_config(page_title="Optimizador Logístico", page_icon="🚚", layout="wide")
ORS_API_KEY = "TU_API_KEY_AQUI" # Dejar vacío si no se tiene
# Configurar cliente ORS solo si existe la key
ors_client = openrouteservice.Client(key=ORS_API_KEY) if ORS_API_KEY else None

# ==========================================
# CLASE 1: GESTIÓN DE DATOS (ETL)
# ==========================================
class DataManager:
    """Encargada de leer, limpiar y estructurar los datos desde Excel."""

    @staticmethod
    def _clean_str(val: Any) -> str:
        return str(val).strip()

    @staticmethod
    def _safe_float(val: Any, default: float = 0.0) -> float:
        try:
            return float(str(val).replace(',', '.'))
        except ValueError:
            return default

    def parse_excel(self, file) -> Tuple[Dict, List[Dict], pd.DataFrame]:
        """
        Lee el archivo Excel y busca las secciones por palabras clave.
        Retorna: (Info CEDI, Lista Flota, DataFrame Pedidos)
        """
        df_raw = pd.read_excel(file, header=None).fillna('')
        
        idx_cedi, idx_flota, idx_pedidos = -1, -1, -1
        
        # Bucle 1: Escaneo inicial para encontrar índices de secciones
        for i, row in df_raw.iterrows():
            row_str = " ".join(row.astype(str)).lower()
            if 'cedi' in row_str and idx_cedi == -1:
                idx_cedi = i
            elif 'caracteristicas de flota' in row_str:
                idx_flota = i
            elif 'caracteristicas de pedidos' in row_str:
                idx_pedidos = i
        
        # Procesar CEDI
        cedi_info = {'name': 'CEDI Central', 'direccion': '', 'lat': None, 'lon': None}
        if idx_cedi != -1:
            # Asumimos que la info está 1 o 2 filas debajo del título
            try:
                row_data = df_raw.iloc[idx_cedi + 1]
                cedi_info['name'] = self._clean_str(row_data[0])
                cedi_info['direccion'] = self._clean_str(row_data[1])
            except IndexError:
                pass

        # Procesar Flota
        fleet_list = []
        if idx_flota != -1:
            # Iteramos desde la fila de flota hasta encontrar una vacía o el inicio de pedidos
            start_row = idx_flota + 1
            limit = idx_pedidos if idx_pedidos != -1 else len(df_raw)
            
            for i in range(start_row, limit):
                row = df_raw.iloc[i]
                # Si la fila está vacía, saltar
                if not "".join(row.astype(str)).strip():
                    continue
                
                # Extraemos datos asumiendo columnas fijas (ajustar según tu Excel real)
                # Formato esperado: Tipo | Cap(kg) | Cap(m3) | Vel(km/h) | Inicio | Fin
                vals = [self._clean_str(x) for x in row if self._clean_str(x) != '']
                if len(vals) >= 4:
                    fleet_list.append({
                        'tipo': vals[0],
                        'capacity_kg': self._safe_float(vals[1]),
                        'capacity_m3': self._safe_float(vals[2]),
                        'speed_kmh': self._safe_float(vals[3], 40.0),
                        'shift_start': vals[4] if len(vals) > 4 else '07:00',
                        'shift_end': vals[5] if len(vals) > 5 else '19:00'
                    })

        # Procesar Pedidos
        df_pedidos = pd.DataFrame()
        if idx_pedidos != -1:
            # Buscar la fila de encabezados real dentro de la sección de pedidos
            header_offset = 0
            for k in range(5): # Buscar en las siguientes 5 filas
                row_vals = df_raw.iloc[idx_pedidos + k].astype(str).str.lower().tolist()
                if 'peso' in row_vals or 'dirección' in row_vals or 'direccion' in row_vals:
                    header_offset = k
                    break
            
            # Recargar solo la sección de pedidos con encabezados correctos
            df_pedidos = pd.read_excel(file, header=idx_pedidos + header_offset)
            df_pedidos.columns = [c.strip().lower() for c in df_pedidos.columns]
            df_pedidos = df_pedidos.dropna(how='all').reset_index(drop=True)

        return cedi_info, fleet_list, df_pedidos

    def geocode_pedidos(self, df: pd.DataFrame) -> List[Dict]:
        """Geocodifica direcciones faltantes en el DataFrame de pedidos."""
        geolocator = Nominatim(user_agent="route_optimizer_app_v1")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.0)
        
        processed_points = []
        
        # Bucle de procesamiento de pedidos
        for idx, row in df.iterrows():
            lat = self._safe_float(row.get('lat', 0))
            lon = self._safe_float(row.get('lon', 0))
            address = self._clean_str(row.get('direccion', ''))
            city = self._clean_str(row.get('ciudad', ''))
            
            # Lógica de Geocodificación (si no hay lat/lon pero hay dirección)
            if (lat == 0 or lon == 0) and address:
                full_address = f"{address}, {city}" if city else address
                try:
                    location = geocode(full_address)
                    if location:
                        lat, lon = location.latitude, location.longitude
                except Exception as e:
                    print(f"Error geocodificando {full_address}: {e}")

            if lat != 0 and lon != 0:
                processed_points.append({
                    'id': str(row.get('id', idx)),
                    'nombre': str(row.get('nombre_pedido', f'Pedido {idx+1}')),
                    'lat': lat,
                    'lon': lon,
                    'peso': self._safe_float(row.get('peso')),
                    'volumen': self._safe_float(row.get('volumen')),
                    'service_time': int(self._safe_float(row.get('service_time', 5))),
                    'tw_start': str(row.get('tw_start', '08:00')),
                    'tw_end': str(row.get('tw_end', '18:00')),
                    'prioridad': str(row.get('prioridad', 'Media'))
                })
                
        return processed_points


# ==========================================
# CLASE 2: MOTOR DE RUTEO (OR-TOOLS)
# ==========================================
class RoutingEngine:
    """Encapsula la lógica de optimización matemática."""

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # Radio tierra km
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return R * c

    def _create_time_matrix(self, locations: List[Dict], speed_kmh: float) -> Tuple[List[List[int]], List[List[float]]]:
        size = len(locations)
        time_matrix = [[0] * size for _ in range(size)]
        dist_matrix = [[0.0] * size for _ in range(size)]
        
        # Bucle doble para matriz de distancias (Simétrica para este ejemplo)
        for i in range(size):
            for j in range(size):
                if i == j: 
                    continue
                dist = self.haversine(locations[i]['lat'], locations[i]['lon'], 
                                      locations[j]['lat'], locations[j]['lon'])
                dist_matrix[i][j] = dist
                # Tiempo en minutos = (distancia / velocidad) * 60
                time_matrix[i][j] = int((dist / max(speed_kmh, 1)) * 60)
        
        return time_matrix, dist_matrix

    def _time_str_to_min(self, t_str: str) -> int:
        try:
            h, m = map(int, t_str.split(':'))
            return h * 60 + m
        except:
            return 0

    def solve_vrptw(self, centro: Dict, pedidos: List[Dict], flota: List[Dict], time_limit: int = 30):
        if not pedidos or not flota:
            return None

        # 1. Preparar Nodos (0 es el depósito)
        nodes = [{'lat': centro['lat'], 'lon': centro['lon'], 'peso': 0, 'volumen': 0, 
                  'service': 0, 'tw_start': '00:00', 'tw_end': '23:59'}] + pedidos
        
        # 2. Matrices
        avg_speed = np.mean([f['speed_kmh'] for f in flota])
        time_mat, dist_mat = self._create_time_matrix(nodes, avg_speed)
        
        # 3. Configuración OR-Tools
        manager = pywrapcp.RoutingIndexManager(len(nodes), len(flota), 0)
        routing = pywrapcp.RoutingModel(manager)

        # Callback de tránsito (Tiempo)
        def time_callback(from_idx, to_idx):
            from_node = manager.IndexToNode(from_idx)
            to_node = manager.IndexToNode(to_idx)
            return time_mat[from_node][to_node] + nodes[to_node]['service']

        transit_idx = routing.RegisterTransitCallback(time_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

        # Dimensión de Tiempo
        routing.AddDimension(transit_idx, 3000, 3000, False, 'Time') # Slack grande para permitir espera
        time_dimension = routing.GetDimensionOrDie('Time')

        # Restricciones de Ventanas de Tiempo
        for i, node in enumerate(nodes):
            index = manager.NodeToIndex(i)
            start = self._time_str_to_min(node['tw_start'])
            end = self._time_str_to_min(node['tw_end'])
            time_dimension.CumulVar(index).SetRange(start, end)

        # Restricciones de Capacidad (Peso y Volumen)
        # Nota: OR-Tools requiere enteros, multiplicamos volumen * 1000
        demands_w = [int(n['peso']) for n in nodes]
        demands_v = [int(n['volumen'] * 1000) for n in nodes]
        
        cap_w = [int(f['capacity_kg']) for f in flota]
        cap_v = [int(f['capacity_m3'] * 1000) for f in flota]

        # Callback Peso
        def demand_w_callback(from_idx):
            return demands_w[manager.IndexToNode(from_idx)]
        
        w_idx = routing.RegisterUnaryTransitCallback(demand_w_callback)
        routing.AddDimensionWithVehicleCapacity(w_idx, 0, cap_w, True, 'CapacityW')

        # Callback Volumen
        def demand_v_callback(from_idx):
            return demands_v[manager.IndexToNode(from_idx)]
        
        v_idx = routing.RegisterUnaryTransitCallback(demand_v_callback)
        routing.AddDimensionWithVehicleCapacity(v_idx, 0, cap_v, True, 'CapacityV')

        # 4. Resolver
        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_params.time_limit.seconds = time_limit

        solution = routing.SolveWithParameters(search_params)

        if not solution:
            return None

        # 5. Extraer Rutas
        final_routes = []
        for vehicle_id in range(len(flota)):
            index = routing.Start(vehicle_id)
            route_coords = []
            route_nodes = []
            total_dist = 0
            
            while not routing.IsEnd(index):
                node_idx = manager.IndexToNode(index)
                route_nodes.append(node_idx)
                route_coords.append([nodes[node_idx]['lon'], nodes[node_idx]['lat']])
                
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                
                # Sumar distancia
                next_node = manager.IndexToNode(index)
                total_dist += dist_mat[node_idx][next_node]

            # Agregar punto final (depósito)
            node_idx = manager.IndexToNode(index)
            route_coords.append([nodes[node_idx]['lon'], nodes[node_idx]['lat']])

            if len(route_nodes) > 1: # Solo rutas con pedidos
                final_routes.append({
                    'vehicle_id': vehicle_id,
                    'vehicle_type': flota[vehicle_id]['tipo'],
                    'coordinates': route_coords,
                    'distance_km': total_dist,
                    'nodes_indices': route_nodes
                })
        
        return final_routes

# ==========================================
# CLASE 3: UTILIDADES DE API Y VISUALIZACIÓN
# ==========================================
class GeoUtils:
    @staticmethod
    def get_ors_route_geometry(coords_list):
        """Obtiene la geometría precisa de carretera usando ORS."""
        if not ors_client or len(coords_list) < 2:
            return None
        try:
            # ORS espera [[lon, lat], [lon, lat]]
            route = ors_client.directions(coordinates=coords_list, profile='driving-car', format='geojson')
            return route['features'][0]
        except Exception:
            return None

# ==========================================
# INTERFAZ PRINCIPAL (MAIN)
# ==========================================
def main():
    # Inicialización de estado
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = DataManager()
    if 'engine' not in st.session_state:
        st.session_state.engine = RoutingEngine()
    if 'cedi' not in st.session_state:
        st.session_state.cedi = None
    if 'pedidos' not in st.session_state:
        st.session_state.pedidos = []
    if 'flota' not in st.session_state:
        st.session_state.flota = []
    if 'routes' not in st.session_state:
        st.session_state.routes = None

    st.title("🚚 Sistema Avanzado de Ruteo")
    
    # --- SIDEBAR: INPUT ---
    with st.sidebar:
        st.header("1. Carga de Datos")
        uploaded_file = st.file_uploader("Subir Plantilla Excel (.xlsx)", type="xlsx")
        
        if uploaded_file:
            try:
                cedi_raw, fleet_raw, pedidos_df = st.session_state.data_manager.parse_excel(uploaded_file)
                
                if not st.session_state.flota: # Cargar solo una vez para permitir edits manuales
                    st.session_state.flota = fleet_raw
                
                if cedi_raw.get('name'):
                    st.success(f"CEDI Detectado: {cedi_raw['name']}")
                    # Intentar geocodificar CEDI si no tiene coords
                    if st.session_state.cedi is None:
                         # Simulación de geocodificación simple para CEDI
                         st.session_state.cedi = {'lat': 4.60971, 'lon': -74.08175} # Default Bogotá
                
                if not pedidos_df.empty and not st.session_state.pedidos:
                    with st.spinner("Geocodificando pedidos... esto puede tardar unos segundos"):
                        st.session_state.pedidos = st.session_state.data_manager.geocode_pedidos(pedidos_df)
                    st.success(f"{len(st.session_state.pedidos)} pedidos procesados.")
            
            except Exception as e:
                st.error(f"Error procesando archivo: {str(e)}")

        st.divider()
        st.header("2. Control de Optimización")
        
        if st.button("🚀 Calcular Rutas Óptimas"):
            if st.session_state.cedi and st.session_state.pedidos and st.session_state.flota:
                with st.spinner("Optimizando rutas con Inteligencia Artificial..."):
                    rutas = st.session_state.engine.solve_vrptw(
                        st.session_state.cedi,
                        st.session_state.pedidos,
                        st.session_state.flota
                    )
                    
                    if rutas:
                        st.session_state.routes = rutas
                        st.success(f"Se encontraron {len(rutas)} rutas óptimas.")
                    else:
                        st.error("No se encontró solución factible. Revisa las restricciones.")
            else:
                st.warning("Faltan datos (CEDI, Pedidos o Flota).")

    # --- MAIN AREA: MAPA Y RESULTADOS ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Visualización Geográfica")
        # Centro del mapa
        center = [st.session_state.cedi['lat'], st.session_state.cedi['lon']] if st.session_state.cedi else [4.60, -74.08]
        m = folium.Map(location=center, zoom_start=12)
        
        # Marcador CEDI
        if st.session_state.cedi:
            folium.Marker(
                [st.session_state.cedi['lat'], st.session_state.cedi['lon']], 
                icon=folium.Icon(color='red', icon='warehouse', prefix='fa'),
                popup="CEDI"
            ).add_to(m)

        # Marcadores Pedidos
        for p in st.session_state.pedidos:
            folium.CircleMarker(
                [p['lat'], p['lon']], radius=5, color='blue', fill=True,
                popup=f"{p['nombre']} ({p['peso']}kg)"
            ).add_to(m)

        # Dibujar Rutas
        if st.session_state.routes:
            colors = ['green', 'orange', 'purple', 'blue', 'darkred']
            for i, ruta in enumerate(st.session_state.routes):
                coords = ruta['coordinates']
                color = colors[i % len(colors)]
                
                # Intentar obtener geometría real de ORS, sino usar líneas rectas
                geo_feature = GeoUtils.get_ors_route_geometry(coords)
                
                if geo_feature:
                    folium.GeoJson(
                        geo_feature, 
                        style_function=lambda x, col=color: {'color': col, 'weight': 4}
                    ).add_to(m)
                else:
                    # Fallback líneas rectas (invertimos lat/lon para Folium Polyline)
                    line_coords = [[lat, lon] for lon, lat in coords]
                    folium.PolyLine(line_coords, color=color, weight=4, opacity=0.8).add_to(m)

        st_folium(m, width="100%", height=600)

    with col2:
        st.subheader("Métricas y Detalles")
        if st.session_state.routes:
            total_km = sum(r['distance_km'] for r in st.session_state.routes)
            st.metric("Distancia Total Estimada", f"{total_km:.2f} km")
            st.metric("Vehículos Utilizados", len(st.session_state.routes))
            
            for r in st.session_state.routes:
                with st.expander(f"🚛 Ruta {r['vehicle_id']+1} ({r['vehicle_type']})"):
                    st.write(f"Distancia: {r['distance_km']:.2f} km")
                    st.write(f"Paradas: {len(r['nodes_indices'])-1}") # -1 por el deposito
        else:
            st.info("Sube datos y calcula para ver resultados.")

if __name__ == "__main__":
    main()
