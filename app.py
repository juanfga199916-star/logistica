# ============================================================
# app.py - Optimización de Rutas VRPTW con Excel de 2 Hojas
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import openrouteservice
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from math import radians, cos, sin, asin, sqrt
import time

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="Optimización de Rutas", layout="wide")

ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImY5MTA5MmE2NzVmZDRhYjBhMTk4YjZiNWNiMWY2YjQzIiwiaCI6Im11cm11cjY0In0="   # opcional, si no tienes ORS se usan líneas rectas
client = openrouteservice.Client(key=ORS_API_KEY) if ORS_API_KEY else None

# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------
if "puntos" not in st.session_state:
    st.session_state.puntos = []
if "centro" not in st.session_state:
    st.session_state.centro = None
if "fleet" not in st.session_state:
    st.session_state.fleet = []
if "map_center" not in st.session_state:
    st.session_state.map_center = [4.5, -75.5]
if "seleccionando_centro" not in st.session_state:
    st.session_state.seleccionando_centro = False
if "route_geojson" not in st.session_state:
    st.session_state.route_geojson = None
if "route_metrics" not in st.session_state:
    st.session_state.route_metrics = None


# ------------------------------------------------------------
# FUNCIONES AUXILIARES
# ------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    return 6371 * 2 * asin(
        sqrt(sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2)
    )

def geocode_address(address, city=None):
    geolocator = Nominatim(user_agent="routing_app")
    query = f"{address}, {city}" if city else address
    try:
        loc = geolocator.geocode(query, timeout=10)
        if loc:
            return loc.latitude, loc.longitude
    except:
        return None, None
    return None, None

def time_to_min(t):
    try:
        h, m = str(t).split(":")
        return int(h)*60 + int(m)
    except:
        return 0

# ------------------------------------------------------------
# LECTURA EXCEL (SIEMPRE 2 HOJAS)
# ------------------------------------------------------------

def load_flota(df):
    df.columns = [c.lower() for c in df.columns]

    required = [
        "tipo_vehiculo","capacidad_kg","capacidad_m3",
        "velocidad_kmh","turno_inicio","turno_fin"
    ]
    for r in required:
        if r not in df.columns:
            raise Exception(f"Falta columna '{r}' en hoja Flota")

    fleet = []
    for _, row in df.iterrows():
        fleet.append({
            "tipo": row["tipo_vehiculo"],
            "capacity_kg": float(str(row["capacidad_kg"]).replace(",", ".")),
            "capacity_m3": float(str(row["capacidad_m3"]).replace(",", ".")),
            "speed_kmh": float(str(row["velocidad_kmh"]).replace(",", ".")),
            "shift_start": row["turno_inicio"],
            "shift_end": row["turno_fin"],
        })
    return fleet


def load_pedidos(df):
    df.columns = [c.lower() for c in df.columns]

    required = [
        "nombre_pedido","peso","volumen","prioridad",
        "tw_start","tw_end","ciudad","direccion"
    ]
    for r in required:
        if r not in df.columns:
            raise Exception(f"Falta columna '{r}' en hoja Pedidos")

    puntos = []
    for _, row in df.iterrows():
        nombre = row["nombre_pedido"]
        peso = float(str(row["peso"]).replace(",", "."))
        volumen = float(str(row["volumen"]).replace(",", "."))
        ciudad = row["ciudad"]
        direccion = row["direccion"]

        lat, lon = geocode_address(direccion, ciudad)
        time.sleep(1)

        if lat and lon:
            puntos.append({
                "nombre": nombre,
                "peso": peso,
                "volumen": volumen,
                "prioridad": row["prioridad"],
                "tw_start": row["tw_start"],
                "tw_end": row["tw_end"],
                "direccion": direccion,
                "lat": lat,
                "lon": lon,
                "service_time": 5
            })

    return puntos


# ------------------------------------------------------------
# VRPTW (OR-Tools)
# ------------------------------------------------------------

def solve_vrptw(centro, puntos, flota):

    nodes = [{
        "lat": centro[0], "lon": centro[1],
        "demand_w": 0, "demand_v": 0,
        "service": 0, "tw_start": 0, "tw_end": 1440
    }]

    for p in puntos:
        nodes.append({
            "lat": p["lat"], "lon": p["lon"],
            "demand_w": p["peso"],
            "demand_v": p["volumen"] * 1000, # escalar m³
            "service": p["service_time"],
            "tw_start": time_to_min(p["tw_start"]),
            "tw_end": time_to_min(p["tw_end"])
        })

    N = len(nodes)
    num_vehicles = len(flota)

    # Matrices tiempo/distancia
    time_matrix = [[0]*N for _ in range(N)]
    dist_matrix = [[0]*N for _ in range(N)]

    for i in range(N):
        for j in range(N):
            km = haversine_km(nodes[i]["lat"], nodes[i]["lon"], nodes[j]["lat"], nodes[j]["lon"])
            dist_matrix[i][j] = km
            time_matrix[i][j] = int((km / flota[0]["speed_kmh"]) * 60)

    # OR-Tools Manager
    manager = pywrapcp.RoutingIndexManager(N, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    # Callback de tiempo
    def time_cb(f_i, t_i):
        ni = manager.IndexToNode(f_i)
        nj = manager.IndexToNode(t_i)
        return time_matrix[ni][nj] + nodes[nj]["service"]

    t_idx = routing.RegisterTransitCallback(time_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(t_idx)

    routing.AddDimension(
        t_idx, 1440, 1440, False, "Time"
    )
    time_dim = routing.GetDimensionOrDie("Time")

    # TW por nodo
    for n in range(N):
        idx = manager.NodeToIndex(n)
        time_dim.CumulVar(idx).SetRange(nodes[n]["tw_start"], nodes[n]["tw_end"])

    # Capacidades
    def cap_w(i):
        return int(nodes[manager.IndexToNode(i)]["demand_w"])

    def cap_v(i):
        return int(nodes[manager.IndexToNode(i)]["demand_v"])

    cap_w_idx = routing.RegisterUnaryTransitCallback(cap_w)
    cap_v_idx = routing.RegisterUnaryTransitCallback(cap_v)

    routing.AddDimensionWithVehicleCapacity(
        cap_w_idx, 0, [int(f["capacity_kg"]) for f in flota], True, "CapW"
    )
    routing.AddDimensionWithVehicleCapacity(
        cap_v_idx, 0, [int(f["capacity_m3"]*1000) for f in flota], True, "CapV"
    )

    # Ejecutar
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.time_limit.seconds = 20
    params.log_search = False

    sol = routing.SolveWithParameters(params)
    if not sol:
        return None, None

    routes = []
    km_tot = 0
    min_tot = 0

    for v in range(num_vehicles):
        idx = routing.Start(v)
        seq = []

        while not routing.IsEnd(idx):
            n = manager.IndexToNode(idx)
            seq.append(n)
            idx = sol.Value(routing.NextVar(idx))

        if len(seq) > 1:
            coords = [[nodes[n]["lon"], nodes[n]["lat"]] for n in seq]
            coords.insert(0, [nodes[0]["lon"], nodes[0]["lat"]])
            coords.append([nodes[0]["lon"], nodes[0]["lat"]])

            routes.append({
                "vehicle": flota[v]["tipo"],
                "coords": coords
            })

    return routes, {}


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------

st.title("🚚 Optimización de Rutas VRPTW — Excel con 2 Hojas")

with st.sidebar:
    st.header("📤 Cargar archivo Excel (2 hojas obligatorias)")
    file = st.file_uploader("Archivo .xlsx", type=["xlsx"])

    if file:
        try:
            xls = pd.ExcelFile(file)

            df_flota = pd.read_excel(xls, "Flota")
            df_ped = pd.read_excel(xls, "Pedidos")

            st.session_state.fleet = load_flota(df_flota)
            st.session_state.puntos = load_pedidos(df_ped)

            st.success("Archivo cargado correctamente.")
            st.info(f"Vehículos cargados: {len(st.session_state.fleet)}")
            st.info(f"Pedidos con coordenadas: {len(st.session_state.puntos)}")

        except Exception as e:
            st.error(f"Error al leer archivo: {e}")

    st.divider()
    st.subheader("Seleccionar Centro")
    if st.button("📍 Elegir centro en el mapa"):
        st.session_state.seleccionando_centro = True

    st.divider()
    if st.button("🚀 Optimizar Rutas"):
        if not st.session_state.centro:
            st.warning("Selecciona primero un centro en el mapa.")
        else:
            r, m = solve_vrptw(st.session_state.centro, st.session_state.puntos, st.session_state.fleet)
            st.session_state.route_geojson = r
            st.success("Optimización completada.")


# ------------------------------------------------------------
# MAPA
# ------------------------------------------------------------

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("🗺️ Mapa")

    m = folium.Map(location=st.session_state.map_center, zoom_start=12)

    # Centro
    if st.session_state.centro:
        folium.Marker(
            st.session_state.centro,
            icon=folium.Icon(color="red", icon="home"),
            popup="Centro"
        ).add_to(m)

    # Pedidos
    for p in st.session_state.puntos:
        folium.Marker(
            [p["lat"], p["lon"]],
            tooltip=p["nombre"],
            popup=f"{p['nombre']}<br>{p['direccion']}"
        ).add_to(m)

    # Rutas
    if st.session_state.route_geojson:
        for r in st.session_state.route_geojson:
            folium.PolyLine(
                r["coords"],
                color="blue",
                weight=4,
                tooltip=r["vehicle"]
            ).add_to(m)

    mapdata = st_folium(m, width="100%", height=650)

    # Al hacer clic en el mapa
    if mapdata and mapdata.get("last_clicked"):
        if st.session_state.seleccionando_centro:
            lat = mapdata["last_clicked"]["lat"]
            lon = mapdata["last_clicked"]["lng"]
            st.session_state.centro = [lat, lon]
            st.session_state.seleccionando_centro = False
            st.success("Centro actualizado.")


with col2:
    st.subheader("📊 KPI")
    st.metric("Pedidos", len(st.session_state.puntos))
    st.metric("Vehículos", len(st.session_state.fleet))

    st.subheader("📋 Flota")
    st.dataframe(pd.DataFrame(st.session_state.fleet))

    st.subheader("📦 Pedidos")
    st.dataframe(pd.DataFrame(st.session_state.puntos))


