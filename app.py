# app.py
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
from typing import List, Dict, Tuple

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Optimización de Rutas (VRPTW) - Streamlit Cloud Ready", layout="wide")
ORS_API_KEY = ""  # <-- Si tienes ORS, pégala aquí. Si no, deja en blanco.
client = openrouteservice.Client(key=ORS_API_KEY) if ORS_API_KEY else None

# Default sample path (environment-specific fallback; tu archivo subido en este entorno)
FALLBACK_SAMPLE_XLSX = "/mnt/data/flota 1.xlsx"

# Color map por tipo de vehículo (puedes extender)
VEHICLE_COLOR_MAP = {
    "Camión": "darkred",
    "Camion": "darkred",
    "Moto": "green",
    "Carro": "blue",
    "Van": "orange",
    "Furgoneta": "orange"
}
DEFAULT_VEHICLE_COLOR = "purple"

# Cost default (unidad monetaria por km)
DEFAULT_COST_PER_KM = 0.5

# ---------------------------
# SESSION STATE inicial
# ---------------------------
if "puntos" not in st.session_state:
    st.session_state.puntos = []
if "fleet" not in st.session_state:
    st.session_state.fleet = []
if "centro" not in st.session_state:
    st.session_state.centro = None
if "map_center" not in st.session_state:
    st.session_state.map_center = [4.60971, -74.08175]  # Bogotá por defecto
if "seleccionando_centro" not in st.session_state:
    st.session_state.seleccionando_centro = False
if "route_geojson" not in st.session_state:
    st.session_state.route_geojson = None
if "route_metrics" not in st.session_state:
    st.session_state.route_metrics = None
if "last_fingerprint" not in st.session_state:
    st.session_state.last_fingerprint = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

# ---------------------------
# UTILIDADES
# ---------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 6371 * 2 * asin(sqrt(a))

@st.cache_data(show_spinner=False)
def geocode_address_cached(address: str, city: str = None) -> Tuple[float, float]:
    """Cacheada para Streamlit Cloud: reduce llamadas repetidas a Nominatim."""
    if not address or (isinstance(address, float) and pd.isna(address)):
        return None, None
    geolocator = Nominatim(user_agent="routing_app_streamlit")
    q = f"{address}, {city}" if city and not pd.isna(city) else address
    try:
        loc = geolocator.geocode(q, timeout=10)
        if loc:
            return float(loc.latitude), float(loc.longitude)
    except Exception:
        return None, None
    return None, None

def time_to_minutes(t):
    try:
        s = str(t).strip()
        if ":" in s:
            h,m = s.split(":")
            return int(h)*60 + int(m)
        else:
            # acepta '7:00' o '07:00' y '7'
            return int(float(s))*60
    except Exception:
        return 0

def fingerprint(centro, puntos, fleet) -> str:
    obj = {
        "centro": centro,
        "puntos": [(p.get("lat"), p.get("lon"), p.get("peso"), p.get("volumen"), p.get("tw_start"), p.get("tw_end")) for p in puntos],
        "fleet": fleet
    }
    return pd.util.hash_pandas_object(pd.Series([str(obj)])).astype(str).iloc[0]

# ---------------------------
# LECTURA HOJAS: FLEET y PEDIDOS (siempre dos hojas)
# ---------------------------
def load_fleet_sheet(df: pd.DataFrame) -> List[Dict]:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    required = ["tipo_vehiculo","capacidad_kg","capacidad_m3","velocidad_kmh","turno_inicio","turno_fin"]
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Hoja 'Flota' necesita la columna: {r}")
    fleet = []
    for _, r in df.iterrows():
        try:
            capkg = float(str(r["capacidad_kg"]).replace(",","."))
        except:
            capkg = 0.0
        try:
            capm3 = float(str(r["capacidad_m3"]).replace(",","."))
        except:
            capm3 = 0.0
        try:
            speed = float(str(r["velocidad_kmh"]).replace(",","."))
        except:
            speed = 40.0
        cost_per_km = float(str(r.get("cost_per_km", DEFAULT_COST_PER_KM)).replace(",",".")) if "cost_per_km" in df.columns else DEFAULT_COST_PER_KM
        fleet.append({
            "tipo": r["tipo_vehiculo"],
            "capacity_kg": capkg,
            "capacity_m3": capm3,
            "speed_kmh": speed,
            "shift_start": r["turno_inicio"],
            "shift_end": r["turno_fin"],
            "cost_per_km": cost_per_km
        })
    return fleet

def load_pedidos_sheet(df: pd.DataFrame) -> List[Dict]:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    required = ["nombre_pedido","peso","volumen","prioridad","tw_start","tw_end","ciudad","direccion"]
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Hoja 'Pedidos' necesita la columna: {r}")
    puntos = []
    # Geocode con cache para evitar llamadas repetidas
    for idx, row in df.iterrows():
        nombre = row["nombre_pedido"]
        try:
            peso = float(str(row["peso"]).replace(",","."))
        except:
            peso = 0.0
        try:
            vol = float(str(row["volumen"]).replace(",","."))
        except:
            vol = 0.0
        prioridad = row["prioridad"]
        tws = row["tw_start"]
        twf = row["tw_end"]
        ciudad = row["ciudad"]
        direccion = row["direccion"]
        # geocodificar (pausa pequeña para ser amable con Nominatim)
        lat, lon = geocode_address_cached(str(direccion), str(ciudad))
        time.sleep(0.5)  # pequeño delay para evitar bloqueos
        if lat is None or lon is None:
            # si no se pudo geocodificar, ignorar o dejar para geocodificación manual
            st.warning(f"No se pudo geocodificar: {nombre} -> {direccion} ({ciudad}). Añádelo manualmente en el mapa si lo deseas.")
            continue
        puntos.append({
            "nombre": nombre,
            "peso": peso,
            "volumen": vol,
            "prioridad": prioridad,
            "tw_start": str(tws),
            "tw_end": str(twf),
            "ciudad": ciudad,
            "direccion": direccion,
            "lat": lat,
            "lon": lon,
            "service_time": int(row.get("service_time", 5) if "service_time" in df.columns else 5)
        })
    return puntos

# ---------------------------
# SOLVER VRPTW (simplificado con OR-Tools)
# - versión optimizada: matrices calculadas una vez, tiempo límite pequeño por defecto
# ---------------------------
def solve_vrptw(centro: List[float], puntos: List[Dict], fleet: List[Dict], time_limit_s: int=20):
    if not centro or not puntos or not fleet:
        return None, None

    # construir nodos: depot + pedidos
    nodes = [{"lat": centro[0], "lon": centro[1], "demand_w": 0, "demand_v": 0, "service": 0,
              "tw_start": 0, "tw_end": 24*60}]
    for p in puntos:
        nodes.append({
            "lat": p["lat"],
            "lon": p["lon"],
            "demand_w": int(round(p.get("peso",0))),
            "demand_v": int(round(p.get("volumen",0)*1000)),
            "service": int(p.get("service_time",5)),
            "tw_start": time_to_minutes(p.get("tw_start","08:00")),
            "tw_end": time_to_minutes(p.get("tw_end","18:00"))
        })

    N = len(nodes)
    num_vehicles = len(fleet)

    # matrices distancia/tiempo (usar promedio de velocidad para estimación)
    avg_speed = np.mean([f.get("speed_kmh",40) for f in fleet])
    dist_matrix = [[0]*N for _ in range(N)]
    time_matrix = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i==j:
                dist_matrix[i][j] = 0.0
                time_matrix[i][j] = 0
            else:
                km = haversine_km(nodes[i]["lat"], nodes[i]["lon"], nodes[j]["lat"], nodes[j]["lon"])
                dist_matrix[i][j] = km
                time_matrix[i][j] = int(round((km / max(avg_speed,1)) * 60))

    # preparar OR-Tools
    manager = pywrapcp.RoutingIndexManager(N, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    # tiempo callback
    def travel_time_cb(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        return time_matrix[from_node][to_node] + nodes[to_node]["service"]
    transit_idx = routing.RegisterTransitCallback(travel_time_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    # dimensión tiempo
    routing.AddDimension(transit_idx, 24*60, 24*60, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    # asignar ventanas
    for node_idx in range(N):
        index = manager.NodeToIndex(node_idx)
        time_dim.CumulVar(index).SetRange(nodes[node_idx]["tw_start"], nodes[node_idx]["tw_end"])

    # capacidades (peso)
    demands_w = [nodes[i]["demand_w"] for i in range(N)]
    def demand_w_cb(from_index):
        return demands_w[manager.IndexToNode(from_index)]
    demand_w_idx = routing.RegisterUnaryTransitCallback(demand_w_cb)
    routing.AddDimensionWithVehicleCapacity(demand_w_idx, 0, [int(round(f["capacity_kg"])) for f in fleet], True, "Weight")

    # capacidades (vol)
    demands_v = [nodes[i]["demand_v"] for i in range(N)]
    def demand_v_cb(from_index):
        return demands_v[manager.IndexToNode(from_index)]
    demand_v_idx = routing.RegisterUnaryTransitCallback(demand_v_cb)
    routing.AddDimensionWithVehicleCapacity(demand_v_idx, 0, [int(round(f["capacity_m3"]*1000)) for f in fleet], True, "Volume")

    # parámetros de búsqueda optimizados para Cloud
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = time_limit_s
    search_parameters.log_search = False

    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        return None, None

    # extraer rutas con su secuencia de coords y métricas
    routes_info = []
    total_km = 0.0
    total_min = 0.0
    for v in range(num_vehicles):
        index = routing.Start(v)
        route_nodes = []
        route_km = 0.0
        route_min = 0.0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route_nodes.append(node)
            next_index = solution.Value(routing.NextVar(index))
            if routing.IsEnd(next_index):
                break
            next_node = manager.IndexToNode(next_index)
            route_km += dist_matrix[node][next_node]
            route_min += time_matrix[node][next_node] + nodes[next_node]["service"]
            index = next_index
        # si no hay visitas, saltar
        if len(route_nodes) <= 1:
            routes_info.append({
                "vehicle_idx": v,
                "vehicle_tipo": fleet[v].get("tipo"),
                "coords": []
            })
            continue
        # construir coords con depot inicio y fin
        seq_coords = [[nodes[0]["lon"], nodes[0]["lat"]]]
        for n in route_nodes:
            if n == 0:
                continue
            seq_coords.append([nodes[n]["lon"], nodes[n]["lat"]])
        seq_coords.append([nodes[0]["lon"], nodes[0]["lat"]])
        routes_info.append({
            "vehicle_idx": v,
            "vehicle_tipo": fleet[v].get("tipo"),
            "coords": seq_coords,
            "distance_km_est": route_km,
            "time_min_est": route_min,
            "cost_est": route_km * fleet[v].get("cost_per_km", DEFAULT_COST_PER_KM)
        })
        total_km += route_km
        total_min += route_min

    metrics = {"distance_km": total_km, "time_min": total_min}
    return routes_info, metrics

# ---------------------------
# REQUEST ORS route (opcional)
# ---------------------------
def request_ors_route(coords_sequence):
    if client is None or not coords_sequence or len(coords_sequence) < 2:
        return None, None
    try:
        r = client.directions(coords_sequence, profile="driving-car", format="geojson")
        feat = r.get("features", [None])[0]
        if not feat:
            return None, None
        segments = feat.get("properties", {}).get("segments", [])
        total_distance = sum(s.get("distance",0) for s in segments)
        total_duration = sum(s.get("duration",0) for s in segments)
        return feat, {"distance_m": total_distance, "duration_s": total_duration}
    except Exception as e:
        st.warning(f"ORS request failed: {e}")
        return None, None

# ---------------------------
# INTERFAZ
# ---------------------------
st.title("🚚 Optimización de Rutas VRPTW — Streamlit Cloud ready")
st.write("Carga un archivo .xlsx con DOS hojas: 'Flota' y 'Pedidos'. El sistema geocodificará direcciones y optimizará rutas (VRPTW).")

with st.sidebar:
    st.header("1) Subir Excel (2 hojas obligatorias)")
    uploaded = st.file_uploader("Sube el .xlsx con hojas 'Flota' y 'Pedidos'", type=["xlsx"])
    # fallback a archivo local (ejemplo)
    use_fallback = False
    if not uploaded:
        if st.button("Usar ejemplo local (fallback)"):
            use_fallback = True

    if uploaded or use_fallback:
        try:
            if uploaded:
                xls = pd.ExcelFile(uploaded)
                st.session_state.uploaded_filename = uploaded.name
            else:
                xls = pd.ExcelFile(FALLBACK_SAMPLE_XLSX)
                st.session_state.uploaded_filename = FALLBACK_SAMPLE_XLSX

            df_flota = pd.read_excel(xls, sheet_name="Flota")
            df_pedidos = pd.read_excel(xls, sheet_name="Pedidos")

            # cargar flota y pedidos
            st.session_state.fleet = load_fleet_sheet(df_flota)
            st.session_state.puntos = load_pedidos_sheet(df_pedidos)

            st.success(f"Archivo cargado. Vehículos: {len(st.session_state.fleet)} — Pedidos geocodificados: {len(st.session_state.puntos)}")
            st.write("Primera filas de cada hoja:")
            st.write("Flota:")
            st.dataframe(pd.DataFrame(st.session_state.fleet))
            st.write("Pedidos (primeras 10):")
            st.dataframe(pd.DataFrame(st.session_state.puntos).head(10))
        except Exception as e:
            st.error(f"Error al leer las hojas 'Flota' y 'Pedidos': {e}")

    st.markdown("---")
    st.header("2) Definir CEDI (centro)")
    if st.button("📍 Seleccionar CEDI en el mapa"):
        st.session_state.seleccionando_centro = True
        st.info("Haz clic en el mapa para definir el centro de distribución (CEDI).")

    st.markdown("---")
    st.header("3) Ajustes y Cálculo")
    time_limit = st.number_input("Time limit OR-Tools (seg)", min_value=5, max_value=120, value=20)
    cost_per_km_default = st.number_input("Costo por km (por defecto si no está en Flota)", value=float(DEFAULT_COST_PER_KM))
    st.markdown("Si la hoja Flota contiene columna `cost_per_km`, esa prevalece por vehículo.")
    st.markdown("---")
    if st.button("🚀 Calcular rutas (VRPTW)"):
        if not st.session_state.centro:
            st.warning("Define primero el CEDI en el mapa.")
        elif not st.session_state.puntos:
            st.warning("No hay pedidos geocodificados.")
        elif not st.session_state.fleet:
            st.warning("No hay flota definida.")
        else:
            # asegurar cost_per_km en fleet
            for v in st.session_state.fleet:
                if "cost_per_km" not in v or v.get("cost_per_km") is None:
                    v["cost_per_km"] = cost_per_km_default
            fp = fingerprint(st.session_state.centro, st.session_state.puntos, st.session_state.fleet)
            if fp == st.session_state.last_fingerprint and st.session_state.route_geojson:
                st.info("La solución ya está calculada y actualizada.")
            else:
                with st.spinner("Resolviendo VRPTW (OR-Tools)..."):
                    routes_info, metrics = solve_vrptw(st.session_state.centro, st.session_state.puntos, st.session_state.fleet, time_limit_s=int(time_limit))
                    if not routes_info:
                        st.error("No se encontró solución factible con las restricciones dadas. Intenta relajar ventanas, aumentar vehículos o capacidad.")
                    else:
                        # solicitar ORS para cada ruta y colorear por tipo vehículo
                        features = []
                        total_distance_m = 0
                        total_duration_s = 0
                        for r in routes_info:
                            coords = r.get("coords", [])
                            if not coords:
                                continue
                            vehicle_type = r.get("vehicle_tipo", "vehiculo")
                            color = VEHICLE_COLOR_MAP.get(vehicle_type, DEFAULT_VEHICLE_COLOR)
                            if client:
                                feat, m = request_ors_route(coords)
                                if feat:
                                    # anotar tipo y color en properties para visualización
                                    feat["properties"] = feat.get("properties", {})
                                    feat["properties"]["vehicle_type"] = vehicle_type
                                    feat["properties"]["color"] = color
                                    features.append(feat)
                                    total_distance_m += m["distance_m"]
                                    total_duration_s += m["duration_s"]
                                else:
                                    ls = {"type":"Feature","properties":{"vehicle_type":vehicle_type,"color":color},"geometry":{"type":"LineString","coordinates":coords}}
                                    features.append(ls)
                            else:
                                ls = {"type":"Feature","properties":{"vehicle_type":vehicle_type,"color":color},"geometry":{"type":"LineString","coordinates":coords}}
                                features.append(ls)
                        if features:
                            fc = {"type":"FeatureCollection","features":features}
                            st.session_state.route_geojson = fc
                            if total_distance_m>0:
                                st.session_state.route_metrics = {"distance_m": total_distance_m, "duration_s": total_duration_s}
                            else:
                                # usar estimaciones de solver
                                st.session_state.route_metrics = {"distance_m": metrics["distance_km"]*1000, "duration_s": metrics["time_min"]*60}
                            st.session_state.last_fingerprint = fp
                            st.success("Rutas calculadas y visualizadas.")
                        else:
                            st.error("No se generaron geometrías de ruta.")

# ---------------------------
# PANTALLA PRINCIPAL - MAPA & KPIs
# ---------------------------
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Mapa")
    map_center = st.session_state.map_center
    m = folium.Map(location=map_center, zoom_start=11)

    # marcador centro
    if st.session_state.centro:
        folium.Marker(st.session_state.centro, icon=folium.Icon(color="red", icon="home"), popup="CEDI (centro)").add_to(m)

    # markers pedidos
    for i,p in enumerate(st.session_state.puntos):
        popup = f"<b>{p.get('nombre')}</b><br>{p.get('direccion','')}<br>Prio: {p.get('prioridad')}<br>Peso: {p.get('peso')} kg<br>Vol: {p.get('volumen')} m3<br>TW: {p.get('tw_start')} - {p.get('tw_end')}"
        folium.Marker([p["lat"], p["lon"]], popup=popup, tooltip=p.get("nombre")).add_to(m)

    # rutas (GeoJSON) con color por propiedad
    if st.session_state.route_geojson:
        for feat in st.session_state.route_geojson.get("features", []):
            color = feat.get("properties", {}).get("color", DEFAULT_VEHICLE_COLOR)
            coords = feat.get("geometry", {}).get("coordinates", [])
            # dibujar PolyLine para estabilidad en Cloud (GeoJson a veces requiere estilos complejos)
            folium.PolyLine(coords, color=color, weight=4, opacity=0.9, tooltip=feat.get("properties",{}).get("vehicle_type","vehicle")).add_to(m)

    map_data = st_folium(m, width="100%", height=650)

    # manejar clicks (seleccionar centro o agregar pedido manual)
    if map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]
        if st.session_state.seleccionando_centro:
            st.session_state.centro = [lat, lon]
            st.session_state.seleccionando_centro = False
            st.success("CEDI definido en el mapa.")
        else:
            # agregar pedido manual (mínimos)
            st.session_state.puntos.append({
                "nombre": f"Punto {len(st.session_state.puntos)+1}",
                "peso": 0.0, "volumen": 0.0, "prioridad": "Media",
                "tw_start": "08:00", "tw_end": "18:00",
                "ciudad": "", "direccion": "", "lat": lat, "lon": lon, "service_time": 5
            })
            st.success("Pedido agregado manualmente.")

with col2:
    st.subheader("KPIs & Flota")
    st.write(f"Pedidos cargados: {len(st.session_state.puntos)}")
    st.write(f"Vehículos en flota: {len(st.session_state.fleet)}")
    if st.session_state.route_metrics:
        dist_km = st.session_state.route_metrics["distance_m"]/1000
        dur_min = st.session_state.route_metrics["duration_s"]/60
        st.metric("Distancia total (km)", f"{dist_km:.2f}")
        st.metric("Tiempo total (min)", f"{dur_min:.0f}")
        # calcular costo total por vehículo (si rutas_info fue creada, estimaciones están dentro de GeoJSON properties o se puede reusar fleet)
        # aquí mostramos un cálculo aproximado usando cost_per_km y distribución proporcional
        if st.session_state.fleet:
            # estimar costo total = distancia_total_km * promedio costo por km (más simple)
            avg_cost = np.mean([v.get("cost_per_km", DEFAULT_COST_PER_KM) for v in st.session_state.fleet])
            st.metric("Costo estimado (total)", f"{dist_km * avg_cost:.2f} (unidad)")
    else:
        st.info("Aún no se han calculado rutas")

    st.markdown("---")
    st.subheader("Flota (detalles)")
    if st.session_state.fleet:
        df_fleet_view = pd.DataFrame(st.session_state.fleet)
        # agregar color para mostrar
        df_fleet_view["color"] = df_fleet_view["tipo"].apply(lambda t: VEHICLE_COLOR_MAP.get(str(t), DEFAULT_VEHICLE_COLOR))
        st.dataframe(df_fleet_view)
    else:
        st.info("No hay flota cargada")

st.subheader("Detalle de Pedidos")
if st.session_state.puntos:
    st.dataframe(pd.DataFrame(st.session_state.puntos), use_container_width=True)
else:
    st.info("No hay pedidos cargados aún")


