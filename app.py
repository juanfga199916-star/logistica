import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from math import radians, sin, cos, asin, sqrt

# ============================================
# CONFIG
# ============================================
st.set_page_config(page_title="Optimizador Logístico", page_icon="🚚", layout="wide")

# ============================================
# UTILIDADES
# ============================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2-lat1)
    dlon = radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*asin(sqrt(a))

def geocode(address):
    geolocator = Nominatim(user_agent="optimizador_rutas_v1")
    g = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    loc = g(address)
    if loc:
        return loc.latitude, loc.longitude
    return None, None

def read_excel_with_two_sheets(file):
    df_flota = pd.read_excel(file, sheet_name="Flota")
    df_ped = pd.read_excel(file, sheet_name="Pedidos")

    df_flota.columns = df_flota.columns.str.lower()
    df_ped.columns = df_ped.columns.str.lower()

    return df_flota, df_ped

def geocode_pedidos(df):
    pedidos = []
    for i, row in df.iterrows():
        lat = row.get("lat", None)
        lon = row.get("lon", None)
        if pd.isna(lat) or pd.isna(lon):
            full = f"{row['direccion']}, {row['ciudad']}"
            lat, lon = geocode(full)

        if lat and lon:
            pedidos.append({
                "id": i,
                "nombre": row.get("nombre_pedido", f"Pedido {i+1}"),
                "lat": lat,
                "lon": lon,
                "peso": float(row.get("peso", 0)),
                "volumen": float(row.get("volumen", 0)),
                "service": int(row.get("service_time", 5)),
                "tw_start": row.get("tw_start", "08:00"),
                "tw_end": row.get("tw_end", "18:00"),
                "prioridad": row.get("prioridad", "Media")
            })
    return pedidos

def str_to_min(t):
    try:
        h, m = map(int, str(t).split(":"))
        return h*60 + m
    except:
        return 0

# ============================================
# VRPTW (OR-Tools)
# ============================================
def solve_vrptw(centro, pedidos, flota, time_limit=20):

    nodes = [{
        "lat": centro["lat"],
        "lon": centro["lon"],
        "peso": 0,
        "volumen": 0,
        "service": 0,
        "tw_start": "00:00",
        "tw_end": "23:59"
    }] + pedidos

    N = len(nodes)
    V = len(flota)

    speeds = [f["speed_kmh"] for f in flota]
    avg_speed = np.mean(speeds)

    # matrices
    time_mat = [[0]*N for _ in range(N)]
    dist_mat = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j: continue
            d = haversine(nodes[i]["lat"], nodes[i]["lon"], nodes[j]["lat"], nodes[j]["lon"])
            dist_mat[i][j] = d
            time_mat[i][j] = int((d / max(avg_speed, 1)) * 60)

    man = pywrapcp.RoutingIndexManager(N, V, 0)
    rout = pywrapcp.RoutingModel(man)

    def time_cb(f, t):
        fn = man.IndexToNode(f)
        tn = man.IndexToNode(t)
        return time_mat[fn][tn] + nodes[tn]["service"]

    t_idx = rout.RegisterTransitCallback(time_cb)
    rout.SetArcCostEvaluatorOfAllVehicles(t_idx)

    rout.AddDimension(t_idx, 600, 600, False, "Time")
    tdim = rout.GetDimensionOrDie("Time")

    for i, n in enumerate(nodes):
        idx = man.NodeToIndex(i)
        tdim.CumulVar(idx).SetRange(str_to_min(n["tw_start"]), str_to_min(n["tw_end"]))

    # CAPACIDAD PESO
    demands_w = [int(n["peso"]) for n in nodes]
    caps_w = [int(f["capacity_kg"]) for f in flota]

    def dem_w(i):
        return demands_w[man.IndexToNode(i)]

    dw = rout.RegisterUnaryTransitCallback(dem_w)
    rout.AddDimensionWithVehicleCapacity(dw, 0, caps_w, True, "CapW")

    # CAPACIDAD VOLUMEN
    demands_v = [int(n["volumen"]*1000) for n in nodes]
    caps_v = [int(f["capacity_m3"]*1000) for f in flota]

    def dem_v(i):
        return demands_v[man.IndexToNode(i)]

    dv = rout.RegisterUnaryTransitCallback(dem_v)
    rout.AddDimensionWithVehicleCapacity(dv, 0, caps_v, True, "CapV")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = time_limit

    sol = rout.SolveWithParameters(params)
    if sol is None:
        return None

    rutas = []
    for v in range(V):
        idx = rout.Start(v)
        coords = []
        total_d = 0
        seq_nodes = []

        while not rout.IsEnd(idx):
            n = man.IndexToNode(idx)
            seq_nodes.append(n)
            coords.append([nodes[n]['lon'], nodes[n]['lat']])
            nxt = sol.Value(rout.NextVar(idx))
            n2 = man.IndexToNode(nxt)
            total_d += dist_mat[n][n2]
            idx = nxt

        coords.append([nodes[0]['lon'], nodes[0]['lat']])
        if len(seq_nodes) > 1:
            rutas.append({
                "vehiculo": flota[v]["tipo"],
                "coords": coords,
                "dist_km": total_d,
                "nodos": seq_nodes
            })

    return rutas

# ============================================
# INTERFAZ
# ============================================
st.title("🚚 OPTIMIZADOR LOGÍSTICO — VRPTW + Costos + Mapa")

# -----------------------------------------
# SIDEBAR
# -----------------------------------------
with st.sidebar:
    st.header("📌 CEDI")
    direccion = st.text_input("Dirección del CEDI")
    ciudad = st.text_input("Ciudad")
    centro = None

    if direccion and ciudad:
        lat, lon = geocode(f"{direccion}, {ciudad}")
        if lat:
            centro = {"lat": lat, "lon": lon}
            st.success("📍 CEDI geocodificado correctamente")
        else:
            st.error("No se pudo geocodificar el CEDI")

    st.divider()
    st.header("📂 Cargar archivo Excel")
    up = st.file_uploader("Excel con hojas: Flota y Pedidos", type=["xlsx"])

    st.divider()
    costo_km = st.number_input("Costo por km ($)", min_value=0.0, value=450.0)

    calcular = st.button("🚀 Optimizar Rutas")

# -----------------------------------------
# MAIN
# -----------------------------------------
if up and centro and calcular:
    flota_df, ped_df = read_excel_with_two_sheets(up)

    flota = [{
        "tipo": row["tipo_vehiculo"],
        "capacity_kg": row["capacidad_kg"],
        "capacity_m3": row["capacidad_m3"],
        "speed_kmh": row["velocidad_kmh"],
        "shift_start": row["turno_inicio"],
        "shift_end": row["turno_fin"]
    } for _, row in flota_df.iterrows()]

    st.subheader("🚛 Flota cargada")
    st.dataframe(flota_df)

    # GEOCODIFICAR PEDIDOS
    pedidos = geocode_pedidos(ped_df)

    st.subheader("📦 Pedidos geocodificados")
    st.dataframe(pd.DataFrame(pedidos))

    rutas = solve_vrptw(centro, pedidos, flota)

    if rutas is None:
        st.error("❌ No se encontró solución factible")
    else:
        st.success(f"Se generaron {len(rutas)} rutas")

        # MAPA
        st.subheader("🗺️ Mapa de Rutas")

        m = folium.Map(location=[centro["lat"], centro["lon"]], zoom_start=11)
        folium.Marker([centro["lat"], centro["lon"]], icon=folium.Icon(color="red"), popup="CEDI").add_to(m)

        colors = ["blue", "green", "orange", "purple", "black", "brown"]

        for i, r in enumerate(rutas):
            color = colors[i % len(colors)]
            folium.PolyLine([[lat, lon] for lon, lat in r["coords"]],
                            color=color, weight=5).add_to(m)

        st_folium(m, width=900, height=600)

        # KPIs
        st.subheader("📊 KPIs")

        total_km = sum(r["dist_km"] for r in rutas)
        total_cost = total_km * costo_km

        st.metric("Distancia total", f"{total_km:.2f} km")
        st.metric("Costo total", f"${total_cost:,.0f}")

        for r in rutas:
            st.write(f"### 🚚 {r['vehiculo']}")
            st.write(f"Distancia: {r['dist_km']:.2f} km — Costo: ${r['dist_km']*costo_km:,.0f}")
