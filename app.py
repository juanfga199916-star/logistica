import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from math import radians, cos, sin, asin, ceil
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Panel de Control de Rutas + Simulador", page_icon="🚚", layout="wide")

# --- ESTADO INICIAL ---
if 'puntos' not in st.session_state: st.session_state.puntos = []
if 'map_center' not in st.session_state: st.session_state.map_center = [3.900, -76.300]
if 'route_metrics' not in st.session_state: st.session_state.route_metrics = None
if 'cedis' not in st.session_state: st.session_state.cedis = []

# --- AUXILIARES ---
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 6371 * 2 * asin(np.sqrt(a))

def time_str_to_minutes(t):
    if isinstance(t, str):
        try:
            h, m = map(int, t.split(':'))
            return h*60 + m
        except:
            return 420
    return 420

def solve_vrptw(centro, puntos, fleet_cfg, num_vehicles_override=None, time_limit_s=5):
    """
    Devuelve:
      - rutas_coords: lista por vehículo de listas de [lat,lon]
      - detalle_por_vehiculo: lista de dicts con keys: vehicle_id, sequence (nodos), distancia_km, carga_kg, tiempo_min
      - metrics: resumen total (distancia_km, num_vehiculos_usados)
    """
    # validar flota
    try:
        num_vehicles = int(fleet_cfg.get('cantidad', 1))
        cap_kg = float(fleet_cfg.get('capacidad_kg', 1000))
        speed_kmh = float(fleet_cfg.get('velocidad_kmh', 40))
        speed_km_min = speed_kmh / 60.0
    except Exception as e:
        st.error(f"Error en la configuración de la flota: {e}")
        return None, None, None

    if num_vehicles_override:
        num_vehicles = int(num_vehicles_override)
    # nodos
    nodes = [{'lat': centro[0], 'lon': centro[1], 'demand': 0, 'service': 0,
              'tw_start': fleet_cfg.get('turno_inicio', '07:00'),
              'tw_end': fleet_cfg.get('turno_fin', '19:00')}]
    for p in puntos:
        nodes.append({
            'lat': p['lat'], 'lon': p['lon'], 'demand': float(p.get('peso', 0) or 0), 'service': int(p.get('service', 15)),
            'tw_start': str(p.get('Tw_Start', '07:00')), 'tw_end': str(p.get('Tw_End', '19:00'))
        })
    N = len(nodes)

    # matrices
    dist_matrix = np.zeros((N, N))
    time_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                km = haversine_km(nodes[i]['lat'], nodes[i]['lon'], nodes[j]['lat'], nodes[j]['lon'])
                dist_matrix[i][j] = km
                time_matrix[i][j] = (km / speed_km_min)
            time_matrix[i][j] += nodes[j]['service']

    # OR-Tools
    manager = pywrapcp.RoutingIndexManager(N, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_idx, to_idx):
        f, t = manager.IndexToNode(from_idx), manager.IndexToNode(to_idx)
        return int(time_matrix[f][t])

    transit_idx = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    # Time dimension
    max_time = int(time_str_to_minutes(fleet_cfg.get('turno_fin', '19:00')) + 60)
    routing.AddDimension(transit_idx, max_time, max_time, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    for node_idx in range(N):
        idx = manager.NodeToIndex(node_idx)
        start = time_str_to_minutes(nodes[node_idx]['tw_start'])
        end = time_str_to_minutes(nodes[node_idx]['tw_end'])
        time_dim.CumulVar(idx).SetRange(start, end)

    # Capacity
    def demand_callback(from_idx):
        node = manager.IndexToNode(from_idx)
        return int(nodes[node]['demand'])
    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, [int(cap_kg)]*num_vehicles, True, "Capacity")

    # solve
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.time_limit.seconds = int(time_limit_s)
    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        return None, None, None

    rutas_coords = []
    detalle_por_vehiculo = []
    distancia_total = 0.0
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        seq_nodes = []
        route_coords = [[nodes[0]['lat'], nodes[0]['lon']]]  # inicia en depósito
        veh_dist = 0.0
        veh_load = 0.0
        veh_time = 0.0
        prev_node = manager.IndexToNode(index)
        while not routing.IsEnd(index):
            node_idx = manager.IndexToNode(index)
            seq_nodes.append(node_idx)
            if node_idx != 0:
                route_coords.append([nodes[node_idx]['lat'], nodes[node_idx]['lon']])
                veh_load += nodes[node_idx]['demand']
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            curr_node = manager.IndexToNode(index)
            veh_dist += dist_matrix[manager.IndexToNode(previous_index)][curr_node]
            veh_time += time_matrix[manager.IndexToNode(previous_index)][curr_node]
        # volver al depósito
        route_coords.append([nodes[0]['lat'], nodes[0]['lon']])
        # sólo considerar vehículos que salieron (tienen al menos un nodo distinto del depósito)
        if len(seq_nodes) > 1 or (len(seq_nodes) == 1 and seq_nodes[0] != 0):
            rutas_coords.append(route_coords)
            distancia_total += veh_dist
            detalle_por_vehiculo.append({
                'vehicle_id': vehicle_id+1,
                'sequence': seq_nodes,
                'distancia_km': veh_dist,
                'carga_kg': veh_load,
                'tiempo_min': veh_time,
                'capacidad_kg': cap_kg
            })
    metrics = {
        "distancia_km": distancia_total,
        "num_vehiculos_usados": len(detalle_por_vehiculo)
    }
    return rutas_coords, detalle_por_vehiculo, metrics

# --- INTERFAZ ---
st.title("🗺️ Optimización + Simulador de Vehículos + Dashboard KPIs")

# SIDEBAR: carga, cedis, costo y simulador
with st.sidebar:
    st.header("Carga y Parámetros")
    file = st.file_uploader("Excel (.xlsx) con 'pedidos' y 'flota'", type=["xlsx"])
    costo_km = st.slider("Costo por kilómetro", min_value=0.0, max_value=20.0, value=1.0, step=0.1)

    st.divider()
    st.header("CEDIS (coordenadas)")
    lat_c = st.number_input("Latitud CEDI", value=3.9, format="%.6f")
    lon_c = st.number_input("Longitud CEDI", value=-76.3, format="%.6f")
    nombre_c = st.text_input("Nombre CEDI (opcional)", value=f"CEDI {len(st.session_state.cedis)+1}")
    if st.button("Agregar CEDI"):
        st.session_state.cedis.append({'lat': float(lat_c), 'lon': float(lon_c), 'nombre': nombre_c})
        st.success("CEDI agregado.")

    if st.session_state.cedis:
        cedi_options = [f"{i+1} - {c['nombre']}" for i,c in enumerate(st.session_state.cedis)]
        idx_cedi = st.selectbox("Selecciona CEDI activo", options=list(range(len(st.session_state.cedis))), format_func=lambda x: cedi_options[x])
    else:
        idx_cedi = None

    st.divider()
    st.header("Simulador de vehículos (What-If)")
    st.write("Selecciona casos a probar (número de vehículos).")
    # opciones fijas 3-6 según tu petición, pero permitimos ampliar si la flota lo exige
    vehicles_to_test = st.multiselect("Probar con...", options=[1,2,3,4,5,6,7,8], default=[3,4,5,6])

    st.caption("Se ejecutará el solver para cada caso seleccionado. Tiempo por caso ~5s.")

    # cargamos df si hay archivo
    df_pedidos = None
    df_flota = None
    selected_fleet = None
    if file:
        try:
            xls = pd.ExcelFile(file)
            pedidos_sheet = next((s for s in xls.sheet_names if 'pedido' in s.lower()), None)
            flota_sheet = next((s for s in xls.sheet_names if 'flota' in s.lower()), None)
            if pedidos_sheet and flota_sheet:
                df_pedidos = pd.read_excel(file, sheet_name=pedidos_sheet)
                df_flota = pd.read_excel(file, sheet_name=flota_sheet)
                df_pedidos.columns = df_pedidos.columns.str.strip()
                df_pedidos = df_pedidos.rename(columns={'Latitud':'lat','Longitud':'lon','Peso (kg)':'peso','Peso':'peso'})
                if 'lat' in df_pedidos.columns and df_pedidos['lat'].mean() > 15:
                    df_pedidos['lat'] = df_pedidos['lat'] / 10
                    df_pedidos['lon'] = df_pedidos['lon'] / 10
                    st.warning("Coordenadas escaladas; corregidas dividiendo por 10.")
                df_flota.columns = df_flota.columns.str.strip().str.lower()
                st.success("Datos cargados.")
                if 'tipo_vehiculo' in df_flota.columns:
                    veh = st.selectbox("Tipo de vehículo (hoja flota)", df_flota['tipo_vehiculo'].unique())
                    selected_fleet = df_flota[df_flota['tipo_vehiculo']==veh].iloc[0].to_dict()
                    selected_fleet.setdefault('cantidad', int(selected_fleet.get('cantidad',1)))
                    selected_fleet.setdefault('capacidad_kg', float(selected_fleet.get('capacidad_kg',1000)))
                    selected_fleet.setdefault('velocidad_kmh', float(selected_fleet.get('velocidad_kmh',40)))
                    selected_fleet.setdefault('turno_inicio', selected_fleet.get('turno_inicio','07:00'))
                    selected_fleet.setdefault('turno_fin', selected_fleet.get('turno_fin','19:00'))
                    st.caption(f"Usando configuración: {selected_fleet.get('cantidad',1)} unidades | cap {selected_fleet.get('capacidad_kg')} kg")
                    st.session_state.puntos = df_pedidos.to_dict('records')
                    if 'lat' in df_pedidos.columns:
                        st.session_state.map_center = [df_pedidos.iloc[0]['lat'], df_pedidos.iloc[0]['lon']]
                else:
                    st.error("La hoja 'flota' debe tener columna tipo_vehiculo.")
            else:
                st.error("Excel debe tener hojas con 'pedidos' y 'flota' en su nombre.")
        except Exception as e:
            st.error(f"Error leyendo Excel: {e}")
    else:
        st.info("Sube un Excel para habilitar simulaciones y flota.")

# --- PANEL PRINCIPAL ---
col_map, col_kpis = st.columns((3,1))

with col_map:
    # Mapa base
    m = folium.Map(location=st.session_state.map_center, zoom_start=11)
    # dibujar pedidos
    for p in st.session_state.puntos:
        folium.CircleMarker(location=[p['lat'], p['lon']], radius=5, color='blue', fill=True,
                            tooltip=f"{p.get('Nombre Pedido', p.get('nombre','Pedido'))} | {p.get('peso',0)}kg").add_to(m)
    # dibujar cedis
    for i,c in enumerate(st.session_state.cedis):
        folium.Marker([c['lat'], c['lon']], tooltip=c['nombre'], icon=folium.Icon(color='green')).add_to(m)
    # heatmap de pedidos
    if st.session_state.puntos:
        heat_points = [[p['lat'], p['lon']] for p in st.session_state.puntos]
        HeatMap(heat_points, radius=12, blur=18, min_opacity=0.3).add_to(m)

    st_map = st_folium(m, height=650, use_container_width=True)

with col_kpis:
    st.subheader("Resumen rápido")
    if st.session_state.route_metrics:
        st.metric("Distancia total (última run)", f"{st.session_state.route_metrics.get('distancia_km',0):.1f} km")
        st.metric("Vehículos usados (última run)", st.session_state.route_metrics.get('num_vehiculos_usados', 0))
    else:
        st.info("Aún no se han calculado rutas (usa el simulador en la barra lateral).")

# --- SIMULACIONES: Ejecutar si el usuario presiona ---
if (file and selected_fleet and st.session_state.puntos and vehicles_to_test) and st.button("Ejecutar simulaciones"):
    centro = None
    if idx_cedi is not None:
        centro = [st.session_state.cedis[idx_cedi]['lat'], st.session_state.cedis[idx_cedi]['lon']]
    else:
        centro = [st.session_state.puntos[0]['lat'], st.session_state.puntos[0]['lon']]

    resultados = []
    detalles_por_escenario = {}  # guardar detalle por escenario (veh->metrics y rutas)
    with st.spinner("Ejecutando simulaciones (cada caso puede tardar algunos segundos)..."):
        for nveh in vehicles_to_test:
            rutas, detalle, metrics = solve_vrptw(centro, st.session_state.puntos, selected_fleet, num_vehicles_override=nveh, time_limit_s=5)
            if metrics is None:
                # solución no encontrada
                resultados.append({
                    'num_vehiculos': nveh,
                    'status': 'No feasible',
                    'distancia_km': None,
                    'costo': None,
                    'carga_total_kg': sum([float(p.get('peso',0) or 0) for p in st.session_state.puntos]),
                    'tiempo_total_min': None,
                    'num_vehiculos_usados': 0
                })
                detalles_por_escenario[nveh] = {'rutas': None, 'detalle': None}
            else:
                distancia = metrics['distancia_km']
                carga_total = sum([float(p.get('peso',0) or 0) for p in st.session_state.puntos])
                tiempo_total = sum([d.get('tiempo_min',0) for d in detalle]) if detalle else 0
                costo_total = distancia * costo_km if distancia is not None else None
                resultados.append({
                    'num_vehiculos': nveh,
                    'status': 'OK',
                    'distancia_km': distancia,
                    'costo': costo_total,
                    'carga_total_kg': carga_total,
                    'tiempo_total_min': tiempo_total,
                    'num_vehiculos_usados': metrics.get('num_vehiculos_usados', 0)
                })
                detalles_por_escenario[nveh] = {'rutas': rutas, 'detalle': detalle}
    # mostrar resultados en tabla
    df_res = pd.DataFrame(resultados)
    st.subheader("Resultados de Simulaciones")
    st.dataframe(df_res)

    # Gráficos comparativos con Plotly
    df_ok = df_res[df_res['status']=='OK'].copy()
    if not df_ok.empty:
        # Distancia por número de vehículos
        fig_dist = px.bar(df_ok, x='num_vehiculos', y='distancia_km', text='distancia_km',
                          labels={'num_vehiculos':'# Vehículos','distancia_km':'Distancia (km)'},
                          title="Distancia total por # de vehículos")
        st.plotly_chart(fig_dist, use_container_width=True)

        # Costo por escenario
        fig_cost = px.bar(df_ok, x='num_vehiculos', y='costo', text='costo',
                          labels={'costo':'Costo total','num_vehiculos':'# Vehículos'},
                          title="Costo estimado por # de vehículos")
        st.plotly_chart(fig_cost, use_container_width=True)

        # Tiempo total por escenario
        fig_time = px.bar(df_ok, x='num_vehiculos', y='tiempo_total_min', text='tiempo_total_min',
                          labels={'tiempo_total_min':'Tiempo total (min)','num_vehiculos':'# Vehículos'},
                          title="Tiempo total estimado por # de vehículos")
        st.plotly_chart(fig_time, use_container_width=True)

    # Permitir seleccionar un escenario para ver KPIs y mapa de detalle
    choice = st.selectbox("Selecciona un escenario para ver detalle", options=[r['num_vehiculos'] for r in resultados], format_func=lambda x: f"{x} vehículos")
    scenario = detalles_por_escenario.get(choice)
    if scenario and scenario['detalle']:
        st.subheader(f"KPIs - Escenario {choice} vehículos")
        detalle = scenario['detalle']
        df_det = pd.DataFrame(detalle)
        # KPIs generales
        distancia_total = df_det['distancia_km'].sum()
        costo_total = distancia_total * costo_km
        # costo por vehículo (promedio)
        costo_por_veh = (df_det['distancia_km'] * costo_km).fillna(0)
        # utilización % por vehículo
        df_det['utilizacion_pct'] = (df_det['carga_kg'] / df_det['capacidad_kg']) * 100
        # mostrar resumen
        colA, colB, colC = st.columns(3)
        colA.metric("Distancia total (km)", f"{distancia_total:.2f}")
        colB.metric("Costo total", f"{costo_total:.2f}")
        colC.metric("Vehículos usados", f"{len(df_det)}")
        # tabla por vehículo
        st.markdown("**Detalle por vehículo**")
        df_show = df_det[['vehicle_id','distancia_km','carga_kg','capacidad_kg','utilizacion_pct','tiempo_min']].copy()
        df_show = df_show.rename(columns={
            'vehicle_id':'Vehículo',
            'distancia_km':'Distancia (km)',
            'carga_kg':'Carga (kg)',
            'capacidad_kg':'Capacidad (kg)',
            'utilizacion_pct':'Utilización (%)',
            'tiempo_min':'Tiempo (min)'
        })
        st.dataframe(df_show.style.format({
            'Distancia (km)': '{:.2f}',
            'Carga (kg)': '{:.1f}',
            'Capacidad (kg)': '{:.1f}',
            'Utilización (%)': '{:.1f}',
            'Tiempo (min)': '{:.0f}'
        }))

        # Gráfico distancia por vehículo
        fig_v = px.bar(df_det, x='vehicle_id', y='distancia_km', labels={'vehicle_id':'Vehículo','distancia_km':'Distancia (km)'}, title="Distancia por vehículo")
        st.plotly_chart(fig_v, use_container_width=True)

        # Mapa con rutas del escenario seleccionado (folium)
        m2 = folium.Map(location=st.session_state.map_center, zoom_start=11)
        # heatmap pedidos
        if st.session_state.puntos:
            HeatMap([[p['lat'], p['lon']] for p in st.session_state.puntos], radius=12, blur=18, min_opacity=0.3).add_to(m2)
        # dibujar CEDI
        if idx_cedi is not None:
            folium.Marker([st.session_state.cedis[idx_cedi]['lat'], st.session_state.cedis[idx_cedi]['lon']], tooltip=st.session_state.cedis[idx_cedi]['nombre'], icon=folium.Icon(color='green')).add_to(m2)
        # dibujar rutas con colores
        colors = ['red','blue','green','orange','purple','black','cadetblue','darkred']
        for i, rcoords in enumerate(scenario['rutas'] or []):
            if rcoords:
                folium.PolyLine(rcoords, color=colors[i % len(colors)], weight=4, tooltip=f"Veh {i+1}").add_to(m2)
        st.subheader("Mapa del escenario seleccionado")
        st_folium(m2, height=600, use_container_width=True)

    else:
        st.info("El escenario seleccionado no tiene solución factible o no se calculó detalle.")

    # guardar última run en session_state para mostrar en el panel izquierdo brevemente
    st.session_state.simulation_results = {'table': df_res, 'details': detalles_por_escenario}
    # también actualizar route_metrics con el mejor escenario (por defecto, el que minimiza costo si hay OK)
    df_ok2 = df_res[df_res['status']=='OK']
    if not df_ok2.empty:
        best = df_ok2.sort_values('costo').iloc[0]
        best_n = int(best['num_vehiculos'])
        # poner ruta del mejor como última run
        best_detail = detalles_por_escenario.get(best_n)
        if best_detail:
            # compilar metrics generales
            total_dist = sum([d['distancia_km'] for d in best_detail['detalle']]) if best_detail['detalle'] else 0
            st.session_state.route_metrics = {'distancia_km': total_dist, 'num_vehiculos_usados': len(best_detail['detalle'])}
    st.success("Simulaciones finalizadas.")

# --- FOOTER / INFO ---
st.markdown("---")
st.caption("Notas: Las simulaciones usan distancia en línea recta (Haversine). El solver tiene un límite corto de tiempo para mantener la UI responsiva. Para rutas por calle, integra OpenRouteService o similar (requiere API key).")

