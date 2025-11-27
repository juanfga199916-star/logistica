import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from math import radians, cos, sin, asin, sqrt, ceil
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

###################################################################################################################3
# CONFIGURACIÓN
###################################################################################################################3

st.set_page_config(page_title="Panel de Control de Rutas", page_icon="🚚", layout="wide")

###################################################################################################################3
#ESTADO INICIAL
###################################################################################################################3

if 'puntos' not in st.session_state: st.session_state.puntos = []
if 'map_center' not in st.session_state: st.session_state.map_center = [3.900, -76.300]
if 'route_metrics' not in st.session_state: st.session_state.route_metrics = None
if 'cedis' not in st.session_state: st.session_state.cedis = []  
    
###################################################################################################################3
#FUNCIONES AUXILIARES
###################################################################################################################3

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
        except:
            return 420
    return 420

def solve_vrptw(centro, puntos, fleet_cfg, num_vehicles_override=None):
    """Resuelve VRPTW con OR-Tools (usando distancias Haversine)."""
    try:
        num_vehicles = int(fleet_cfg.get('cantidad', 1))
        cap_kg = float(fleet_cfg.get('capacidad_kg', 1000))
        speed_kmh = float(fleet_cfg.get('velocidad_kmh', 40))
        speed_km_min = speed_kmh / 60.0
    except Exception as e:
        st.error(f"Error en la configuración de la flota: {e}")
        return None, None

    if num_vehicles_override:
        num_vehicles = int(num_vehicles_override)

    nodes = [{'lat': centro[0], 'lon': centro[1], 'demand': 0, 'service': 0,
              'tw_start': fleet_cfg.get('turno_inicio', '07:00'),
              'tw_end': fleet_cfg.get('turno_fin', '19:00')}]

    for p in puntos:
        nodes.append({
            'lat': p['lat'], 'lon': p['lon'], 'demand': p.get('peso', 0), 'service': int(p.get('service', 15)),
            'tw_start': str(p.get('Tw_Start', '07:00')), 'tw_end': str(p.get('Tw_End', '19:00'))
        })

    N = len(nodes)
    dist_matrix = np.zeros((N, N))
    time_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i != j:
                km = haversine_km(nodes[i]['lat'], nodes[i]['lon'], nodes[j]['lat'], nodes[j]['lon'])
                dist_matrix[i][j] = km
                time_matrix[i][j] = km / speed_km_min
            time_matrix[i][j] += nodes[j]['service']

    manager = pywrapcp.RoutingIndexManager(N, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_idx, to_idx):
        f, t = manager.IndexToNode(from_idx), manager.IndexToNode(to_idx)
        return int(time_matrix[f][t])

    transit_idx = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    max_time = int(time_str_to_minutes(fleet_cfg.get('turno_fin', '19:00')) + 60)
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
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, [int(cap_kg)]*num_vehicles, True, "Capacity")

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.time_limit.seconds = 10

    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        return None, None

    rutas_coords = []
    distancia_total = 0.0
    vehiculos_usados = []
    distancias_rutas = []

    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route = [[nodes[0]['lat'], nodes[0]['lon']]]
        used = False
        distancia_ruta = 0.0

        while not routing.IsEnd(index):
            node_idx = manager.IndexToNode(index)
            if node_idx != 0:
                route.append([nodes[node_idx]['lat'], nodes[node_idx]['lon']])
                used = True

            previous_index = index
            index = solution.Value(routing.NextVar(index))
            curr_node = manager.IndexToNode(index)

            distancia_total += dist_matrix[manager.IndexToNode(previous_index)][curr_node]
            distancia_ruta += dist_matrix[manager.IndexToNode(previous_index)][curr_node]

        route.append([nodes[0]['lat'], nodes[0]['lon']])

        if used:
            rutas_coords.append(route)
            vehiculos_usados.append(f"{vehicle_id+1}")
            distancias_rutas.append(distancia_ruta)

    metrics = {
        "distancia_km": distancia_total,
        "vehiculos_usados": vehiculos_usados,
        "num_vehiculos_usados": len(vehiculos_usados),
        "distancias_por_ruta": distancias_rutas
    }
    return rutas_coords, metrics
    
###################################################################################################################3
#INTERFAZ
###################################################################################################################3

st.title("🗺️ Optimización Logística: Pedidos y Flota (Con CEDIS por Coordenadas)")

with st.sidebar:
    st.header("📂 1. Cargar Datos")
    st.info("Sube un Excel con hojas: 'pedidos' y 'flota'.")
    file = st.file_uploader("Archivo Excel (.xlsx)", type=["xlsx"])

    costo_km = st.slider("Costo por kilómetro", 0.0, 10.0, 1.0, 0.1)

    st.divider()
    st.header("🏬 2. CEDIS (coordenadas)")

    cedi_lat = st.number_input("Latitud CEDI", value=3.9)
    cedi_lon = st.number_input("Longitud CEDI", value=-76.3)
    cedi_nombre = st.text_input("Nombre CEDI", value=f"CEDI {len(st.session_state.cedis)+1}")

    if st.button("➕ Agregar CEDI"):
        st.session_state.cedis.append({'lat': float(cedi_lat), 'lon': float(cedi_lon), 'nombre': cedi_nombre})
        st.success(f"CEDI '{cedi_nombre}' agregado.")

    if st.session_state.cedis:
        nombres = [f"{i+1} - {c['nombre']}" for i,c in enumerate(st.session_state.cedis)]
        seleccion_cedi_idx = st.selectbox("Selecciona CEDI activo",
                                          options=list(range(len(st.session_state.cedis))),
                                          format_func=lambda x: nombres[x])
    else:
        seleccion_cedi_idx = None

    st.divider()
    st.header("📦 3. Flota")

    df_pedidos, df_flota, selected_fleet = None, None, None
    if file:
        try:
            xls = pd.ExcelFile(file)
            pedidos_sheet = next((s for s in xls.sheet_names if 'pedido' in s.lower()), None)
            flota_sheet = next((s for s in xls.sheet_names if 'flota' in s.lower()), None)

            if pedidos_sheet and flota_sheet:
                df_pedidos = pd.read_excel(file, sheet_name=pedidos_sheet)
                df_flota = pd.read_excel(file, sheet_name=flota_sheet)

                df_pedidos.columns = df_pedidos.columns.str.strip()
                df_pedidos = df_pedidos.rename(columns={'Latitud':'lat', 'Longitud':'lon', 'Peso (kg)':'peso', 'Peso':'peso'})

                if 'lat' in df_pedidos.columns and df_pedidos['lat'].mean() > 15:
                    df_pedidos['lat'] /= 10
                    df_pedidos['lon'] /= 10
                    st.warning("Detecté coordenadas mal escaladas. Ajustadas ÷10.")

                df_flota.columns = df_flota.columns.str.strip().str.lower()

                if 'tipo_vehiculo' in df_flota.columns:
                    tipo_veh = st.selectbox("Tipo de vehículo", df_flota['tipo_vehiculo'].unique())
                    selected_fleet = df_flota[df_flota['tipo_vehiculo'] == tipo_veh].iloc[0].to_dict()

                    selected_fleet.setdefault('cantidad', 1)
                    selected_fleet.setdefault('capacidad_kg', 1000)
                    selected_fleet.setdefault('velocidad_kmh', 40)
                    selected_fleet.setdefault('turno_inicio', '07:00')
                    selected_fleet.setdefault('turno_fin', '19:00')

                    st.session_state.puntos = df_pedidos.to_dict('records')
                    st.session_state.map_center = [df_pedidos.iloc[0]['lat'], df_pedidos.iloc[0]['lon']]
                else:
                    st.error("La hoja de flota debe incluir 'tipo_vehiculo'.")

            else:
                st.error("El archivo debe tener hojas con 'pedido' y 'flota'.")

        except Exception as e:
            st.error(f"Error procesando archivo: {e}")

col1, col2 = st.columns((3,1))

with col1:

    m = folium.Map(location=st.session_state.map_center, zoom_start=11)

    for i, c in enumerate(st.session_state.cedis):
        folium.Marker(
            location=[c['lat'], c['lon']],
            tooltip=f"CEDI {i+1}: {c['nombre']}",
            icon=folium.Icon(color='darkgreen', icon='warehouse')
        ).add_to(m)

    for p in st.session_state.puntos:
        folium.CircleMarker(
            location=[p['lat'], p['lon']], radius=5, color="blue", fill=True,
            tooltip=f"{p.get('Nombre Pedido', p.get('nombre','Pedido'))} | {p.get('peso',0)}kg"
        ).add_to(m)

    if seleccion_cedi_idx is not None:
        cedi_act = st.session_state.cedis[seleccion_cedi_idx]
        for p in st.session_state.puntos:
            folium.PolyLine(
                [[cedi_act['lat'], cedi_act['lon']], [p['lat'], p['lon']]],
                color='gray', weight=1.5, dash_array='5'
            ).add_to(m)

    if selected_fleet and st.button("🚀 Calcular Rutas & Costos"):
        if seleccion_cedi_idx is not None:
            centro = [
                st.session_state.cedis[seleccion_cedi_idx]['lat'],
                st.session_state.cedis[seleccion_cedi_idx]['lon']
            ]
        else:
            centro = [st.session_state.puntos[0]['lat'], st.session_state.puntos[0]['lon']]

        total_demand = sum([float(p.get('peso', 0) or 0) for p in st.session_state.puntos])
        cap_kg = float(selected_fleet.get('capacidad_kg', 1000))
        num_necesarios = max(1, ceil(total_demand / cap_kg))

        rutas, metricas = solve_vrptw(
            centro, st.session_state.puntos, selected_fleet,
            num_vehicles_override=num_necesarios
        )

        if rutas:
            colors = ['red','green','blue','orange','purple','black','cadetblue','darkred']

            for i, ruta in enumerate(rutas):
                if len(ruta) > 1:
                    folium.PolyLine(
                        ruta, weight=4, color=colors[i % len(colors)], opacity=0.8,
                        tooltip=f"Ruta {i+1}"
                    ).add_to(m)
                    
###################################################################################################################
#LEYENDA DE GRAFICOS
###################################################################################################################

            legend_html = """
            <div style="
                position: fixed;
                bottom: 30px;
                left: 30px;
                z-index: 9999;
                background-color: white;
                padding: 10px;
                border-radius: 8px;
                box-shadow: 0 0 8px rgba(0,0,0,0.3);
                font-size: 14px;
            ">
            <b>Colores de Rutas</b><br>
            """

            for i in range(len(rutas)):
                color = colors[i % len(colors)]
                legend_html += f"""
                    <div style="display:flex;align-items:center;margin-top:4px;">
                        <div style="width:15px;height:15px;background:{color};
                                    margin-right:6px;border:1px solid #333;"></div>
                        Ruta {i+1}
                    </div>
                """

            legend_html += "</div>"
            m.get_root().html.add_child(folium.Element(legend_html))
            
            st.session_state.route_metrics = metricas
            costo_total = metricas['distancia_km'] * costo_km

            st.success("Rutas optimizadas con éxito.")
            st.info(
                f"Vehículos usados: {metricas['num_vehiculos_usados']} — "
                f"Distancia total: {metricas['distancia_km']:.2f} km — "
                f"Costo: {costo_total:.2f}"
            )
        else:
            st.session_state.route_metrics = None
            st.error("No se encontró solución factible.")

    st_folium(m, height=650, use_container_width=True)

with col2:

    st.subheader("📋 Resumen / Métricas")

    if st.session_state.route_metrics:
        dist = st.session_state.route_metrics['distancia_km']
        costo_total = dist * costo_km

        st.metric("Distancia Total Estimada", f"{dist:.1f} km")
        st.metric("Costo Estimado", f"{costo_total:.2f}")

        st.write("Vehículos usados:", ", ".join(st.session_state.route_metrics["vehiculos_usados"]))

        st.subheader("📌 Costos por Ruta")
        for i, d in enumerate(st.session_state.route_metrics["distancias_por_ruta"], start=1):
            st.write(f"Ruta {i}: {d:.2f} km — Costo: {(d * costo_km):.2f}")

    else:
        st.info("Aún no hay rutas calculadas.")

    if selected_fleet:
        st.markdown("**Configuración flota:**")
        st.write(selected_fleet)

    with st.expander("Ver Pedidos Cargados"):
        if df_pedidos is not None:
            st.dataframe(df_pedidos)

    with st.expander("CEDIS registrados"):
        if st.session_state.cedis:
            st.dataframe(pd.DataFrame(st.session_state.cedis))

    st.caption("Distancias calculadas por Haversine (línea recta).")
