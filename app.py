




import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from math import radians, cos, sin, asin, sqrt, ceil
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# ===============================
#    CONFIGURACIÓN STREAMLIT
# ===============================
st.set_page_config(page_title="Panel de Control de Rutas", page_icon="🚚", layout="wide")

# ===============================
#    ESTADOS DE SESIÓN
# ===============================
if 'puntos' not in st.session_state:
    st.session_state.puntos = []

if 'map_center' not in st.session_state:
    st.session_state.map_center = [3.9, -76.3]

if 'route_metrics' not in st.session_state:
    st.session_state.route_metrics = None

if 'cedis' not in st.session_state:
    st.session_state.cedis = []


# ===============================
#        FUNCIONES
# ===============================
def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia Haversine en km."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    return 6371 * 2 * asin(sqrt(a))


def time_str_to_minutes(t):
    """Convierte HH:MM en minutos."""
    if isinstance(t, str):
        try:
            h, m = map(int, t.split(":"))
            return h * 60 + m
        except:
            return 420
    return 420


def solve_vrptw(centro, puntos, fleet_cfg, num_vehicles_override=None):
    """Resuelve VRP con TW usando OR-Tools."""
    try:
        num_vehicles = int(fleet_cfg.get("cantidad", 1))
        cap_kg = float(fleet_cfg.get("capacidad_kg", 1000))
        speed = float(fleet_cfg.get("velocidad_kmh", 40)) / 60
    except Exception as e:
        st.error(f"Error en configuración de flota: {e}")
        return None, None

    if num_vehicles_override:
        num_vehicles = num_vehicles_override

    # Construcción de nodos
    nodes = [{
        "lat": centro[0],
        "lon": centro[1],
        "demand": 0,
        "service": 0,
        "tw_start": fleet_cfg.get("turno_inicio", "07:00"),
        "tw_end": fleet_cfg.get("turno_fin", "19:00")
    }]

    for p in puntos:
        nodes.append({
            "lat": p["lat"],
            "lon": p["lon"],
            "demand": p.get("peso", 0),
            "service": int(p.get("service", 10)),
            "tw_start": str(p.get("Tw_Start", "07:00")),
            "tw_end": str(p.get("Tw_End", "19:00"))
        })

    N = len(nodes)

    # Matrices de distancia y tiempo
    dist_matrix = np.zeros((N, N))
    time_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i != j:
                km = haversine_km(nodes[i]["lat"], nodes[i]["lon"], nodes[j]["lat"], nodes[j]["lon"])
                dist_matrix[i][j] = km
                time_matrix[i][j] = km / speed
            time_matrix[i][j] += nodes[j]["service"]

    manager = pywrapcp.RoutingIndexManager(N, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_idx, to_idx):
        f = manager.IndexToNode(from_idx)
        t = manager.IndexToNode(to_idx)
        return int(time_matrix[f][t])

    transit_idx = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    max_time = time_str_to_minutes(fleet_cfg.get("turno_fin", "19:00")) + 120
    routing.AddDimension(transit_idx, max_time, max_time, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    for node_idx in range(N):
        idx = manager.NodeToIndex(node_idx)
        time_dim.CumulVar(idx).SetRange(
            time_str_to_minutes(nodes[node_idx]["tw_start"]),
            time_str_to_minutes(nodes[node_idx]["tw_end"])
        )

    def demand_callback(idx):
        return int(nodes[manager.IndexToNode(idx)]["demand"])

    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, [cap_kg] * num_vehicles, True, "Capacity")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.time_limit.seconds = 10
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(params)
    if not solution:
        return None, None

    rutas = []
    distancias_rutas = []
    distancia_total = 0
    usados = []

    for v in range(num_vehicles):
        index = routing.Start(v)
        route_coords = [[nodes[0]["lat"], nodes[0]["lon"]]]
        used = False
        dist_ruta = 0

        while True:
            next_index = solution.Value(routing.NextVar(index))
            if routing.IsEnd(next_index):
                break

            ni = manager.IndexToNode(next_index)
            route_coords.append([nodes[ni]["lat"], nodes[ni]["lon"]])
            used = True

            dist_ruta += dist_matrix[manager.IndexToNode(index)][ni]
            index = next_index

        route_coords.append([nodes[0]["lat"], nodes[0]["lon"]])

        if used:
            rutas.append(route_coords)
            usados.append(str(v + 1))
            distancias_rutas.append(dist_ruta)
            distancia_total += dist_ruta

    metrics = {
        "distancia_km": distancia_total,
        "distancias_por_ruta": distancias_rutas,
        "vehiculos_usados": usados,
        "num_vehiculos_usados": len(usados)
    }

    return rutas, metrics


# ===============================
#    INTERFAZ PRINCIPAL
# ===============================
st.title("🗺️ Optimización Logística con VRP & Ventanas de Tiempo")

with st.sidebar:
    st.header("📂 Cargar archivo")
    file = st.file_uploader("Excel con hojas: pedidos / flota", type=["xlsx"])

    costo_km = st.slider("Costo por km", 0.0, 10.0, 1.0, 0.1)

    st.divider()
    st.header("🏬 CEDIS")

    lat_c = st.number_input("Latitud CEDI", value=3.9)
    lon_c = st.number_input("Longitud CEDI", value=-76.3)
    nombre_c = st.text_input("Nombre", value="CEDI 1")

    if st.button("Agregar CEDI"):
        st.session_state.cedis.append({"lat": lat_c, "lon": lon_c, "nombre": nombre_c})
        st.success("CEDI agregado ✔️")

    if st.session_state.cedis:
        seleccion_cedi = st.selectbox(
            "Seleccionar CEDI",
            range(len(st.session_state.cedis)),
            format_func=lambda i: st.session_state.cedis[i]["nombre"]
        )
    else:
        seleccion_cedi = None

    st.divider()
    st.header("🚚 Flota")

    df_pedidos = df_flota = selected_fleet = None

    if file:
        try:
            xls = pd.ExcelFile(file)
            ped_sheet = next(s for s in xls.sheet_names if "pedido" in s.lower())
            flo_sheet = next(s for s in xls.sheet_names if "flota" in s.lower())

            df_pedidos = pd.read_excel(file, sheet_name=ped_sheet)
            df_flota = pd.read_excel(file, sheet_name=flo_sheet)

            df_pedidos.columns = df_pedidos.columns.str.strip()
            df_pedidos = df_pedidos.rename(columns={"Latitud": "lat", "Longitud": "lon", "Peso": "peso"})

            st.session_state.puntos = df_pedidos.to_dict("records")
            st.session_state.map_center = [df_pedidos.iloc[0]["lat"], df_pedidos.iloc[0]["lon"]]

            tipo = st.selectbox("Tipo de vehículo", df_flota["tipo_vehiculo"].unique())
            selected_fleet = df_flota[df_flota["tipo_vehiculo"] == tipo].iloc[0].to_dict()

            selected_fleet.setdefault("velocidad_kmh", 40)
            selected_fleet.setdefault("capacidad_kg", 1000)
            selected_fleet.setdefault("cantidad", 1)
            selected_fleet.setdefault("turno_inicio", "07:00")
            selected_fleet.setdefault("turno_fin", "19:00")

        except Exception as e:
            st.error(f"Error leyendo archivo: {e}")


# ===============================
#           MAPA
# ===============================
col1, col2 = st.columns([3, 1])

with col1:

    m = folium.Map(location=st.session_state.map_center, zoom_start=12)

    # ------- Mostrar CEDIS -------
    for i, c in enumerate(st.session_state.cedis):
        folium.Marker([c["lat"], c["lon"]], tooltip=c["nombre"],
                      icon=folium.Icon(color="green")).add_to(m)

    # ------- Mostrar pedidos ------
    for p in st.session_state.puntos:
        folium.CircleMarker([p["lat"], p["lon"]], radius=4,
                            color="blue", fill=True).add_to(m)

    # ------- Cálculo de rutas -------
    if selected_fleet and st.button("🚀 Calcular Rutas"):
        cedi = st.session_state.cedis[seleccion_cedi]
        centro = [cedi["lat"], cedi["lon"]]

        total_demand = sum([p.get("peso", 0) for p in st.session_state.puntos])
        cap = selected_fleet.get("capacidad_kg", 1000)
        veh_necesarios = max(1, ceil(total_demand / cap))

        rutas, metricas = solve_vrptw(centro, st.session_state.puntos, selected_fleet,
                                      veh_necesarios)

        if rutas:
            colores = ["red", "green", "blue", "purple", "orange", "black"]

            for i, ruta in enumerate(rutas):
                folium.PolyLine(ruta, color=colores[i % len(colores)], weight=4).add_to(m)

            st.session_state.route_metrics = metricas
            st.success("Rutas generadas correctamente ✔️")

        else:
            st.error("No fue posible generar rutas.")

    st_folium(m, height=650, use_container_width=True)

with col2:

    st.subheader("📊 Métricas")

    if st.session_state.route_metrics:
        m = st.session_state.route_metrics
        costo = m["distancia_km"] * costo_km

        st.metric("Distancia total", f"{m['distancia_km']:.2f} km")
        st.metric("Costo total estimado", f"${costo:,.2f}")

        for i, d in enumerate(m["distancias_por_ruta"], 1):
            st.write(f"Ruta {i}: {d:.2f} km — Costo: ${d * costo_km:.2f}")

    if df_pedidos is not None:
        with st.expander("Ver pedidos"):
            st.dataframe(df_pedidos)

