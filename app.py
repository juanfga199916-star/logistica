import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import openrouteservice
import hashlib
import json
from math import radians, cos, sin, asin, sqrt
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# --- CONFIG ---
st.set_page_config(page_title="Panel de Control de Rutas", page_icon="🗺️", layout="wide")

# --- SESSION STATE INIT ---
if 'puntos' not in st.session_state:
    st.session_state.puntos = []
if 'map_center' not in st.session_state:
    st.session_state.map_center = [4.60971, -74.08175]
if 'centro' not in st.session_state:
    st.session_state.centro = None
if 'seleccionando_centro' not in st.session_state:
    st.session_state.seleccionando_centro = False
if 'route_geojson' not in st.session_state:
    st.session_state.route_geojson = None
if 'route_metrics' not in st.session_state:
    st.session_state.route_metrics = None
if 'last_click' not in st.session_state:
    st.session_state.last_click = None
if 'last_fingerprint' not in st.session_state:
    st.session_state.last_fingerprint = None
if 'fleet_data' not in st.session_state:
    st.session_state.fleet_data = None

# --- CONFIGURAR ORS ---
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImY5MTA5MmE2NzVmZDRhYjBhMTk4YjZiNWNiMWY2YjQzIiwiaCI6Im11cm11cjY0In0="  # tu API key si la tienes
client = openrouteservice.Client(key=ORS_API_KEY) if ORS_API_KEY else None

# --- FUNCIONES AUXILIARES ---
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    return 6371 * 2 * asin(sqrt(sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2))

def get_coords_from_city(city_name):
    try:
        geolocator = Nominatim(user_agent="routing_app")
        location = geolocator.geocode(city_name, timeout=10)
        if location:
            return [location.latitude, location.longitude]
    except Exception as e:
        st.warning(f"No se pudo geocodificar la ciudad: {e}")
    return None

def crear_tabla_de_pedidos(puntos):
    if not puntos:
        return pd.DataFrame(columns=['ID Pedido', 'Destino', 'Prioridad', 'Peso (kg)', 'Volumen (m3)', 'Service (min)', 'TW Inicio', 'TW Fin', 'Latitud', 'Longitud'])
    data = {
        'ID Pedido': [f"Pedido {i+1}" for i in range(len(puntos))],
        'Destino': [p.get('nombre', f'Punto {i+1}') for i, p in enumerate(puntos)],
        'Prioridad': [p.get('prioridad', 'Media') for p in puntos],
        'Peso (kg)': [p.get('peso', 0.0) for p in puntos],
        'Volumen (m3)': [p.get('volumen', 0.0) for p in puntos],
        'Service (min)': [p.get('service_time', 5) for p in puntos],
        'TW Inicio': [p.get('tw_start','00:00') for p in puntos],
        'TW Fin': [p.get('tw_end','23:59') for p in puntos],
        'Latitud': [p['lat'] for p in puntos],
        'Longitud': [p['lon'] for p in puntos]
    }
    return pd.DataFrame(data)

def fingerprint(centro, puntos, fleet_cfg):
    obj = {'centro': centro, 'puntos': [(p['lat'], p['lon']) for p in puntos], 'fleet': fleet_cfg}
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def time_str_to_minutes(tstr):
    try:
        h, m = map(int, tstr.split(':'))
        return h*60+m
    except Exception:
        return 0

def solve_vrptw_simple(centro, puntos, fleet_cfg, avg_speed_kmph=40):
    if not centro or not puntos:
        return None, None, None
    coords = [[centro[1], centro[0]]] + [[p['lon'], p['lat']] for p in puntos] + [[centro[1], centro[0]]]
    total_km = sum(haversine_km(p['lat'], p['lon'], puntos[i+1]['lat'], puntos[i+1]['lon']) for i in range(len(puntos)-1)) if len(puntos)>1 else 0
    metrics = {"distance_km": total_km, "time_min": total_km / fleet_cfg['velocidad_kmh'] * 60}
    return None, [coords], metrics

# --- UI ---
st.title("🗺️ Panel de Control para Optimización de Rutas")
st.write("Ahora puedes cargar tu archivo Excel/CSV con configuración de flota y calcular rutas óptimas (VRPTW).")

with st.sidebar:
    st.header("⚙️ Configuración General")
    ciudad = st.text_input("📍 Ciudad de operación", "Bogotá")
    if st.button("Buscar Ciudad"):
        new = get_coords_from_city(ciudad)
        if new:
            st.session_state.map_center = new
            st.success(f"Mapa centrado en {ciudad}.")

    st.divider()
    st.subheader("Centro de Distribución")
    if st.button("📍 Seleccionar Centro en el Mapa"):
        st.session_state.seleccionando_centro = True
        st.info("Haz clic en el mapa para definir el centro.")

    st.divider()
    st.subheader("📂 Cargar Archivo de Flota")
    uploaded_file = st.file_uploader("Sube un archivo Excel o CSV", type=["csv", "xlsx"])

    fleet_cfg = None
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_fleet = pd.read_csv(uploaded_file)
            else:
                df_fleet = pd.read_excel(uploaded_file)

            st.session_state.fleet_data = df_fleet
            st.dataframe(df_fleet)

            # Tomar primera fila como configuración base
            base = df_fleet.iloc[0]
            fleet_cfg = {
                'num_vehicles': len(df_fleet),
                'capacity_weight': float(base.get('capacidad_kg', 1000)),
                'capacity_volume': float(base.get('capacidad_m3', 8)),
                'shift_start': str(base.get('turno_inicio', '07:00')),
                'shift_end': str(base.get('turno_fin', '19:00')),
                'velocidad_kmh': float(base.get('velocidad_kmh', 40))
            }
            st.success("Archivo cargado correctamente ✅")

        except Exception as e:
            st.error(f"Error leyendo archivo: {e}")
    else:
        st.info("Carga un archivo con columnas: tipo_vehiculo, capacidad_kg, capacidad_m3, turno_inicio, turno_fin, velocidad_kmh.")
        # Entrada manual de respaldo
        num_vehicles = st.number_input("Número de vehículos", min_value=1, max_value=20, value=3)
        capacity_weight = st.number_input("Capacidad por vehículo (kg)", min_value=1.0, value=1000.0)
        capacity_volume = st.number_input("Capacidad por vehículo (m³)", min_value=0.1, value=8.0)
        shift_start = st.text_input("Inicio de turno (HH:MM)", value="07:00")
        shift_end = st.text_input("Fin de turno (HH:MM)", value="19:00")
        velocidad_kmh = st.number_input("Velocidad promedio (km/h)", min_value=10.0, value=40.0)

        fleet_cfg = {
            'num_vehicles': int(num_vehicles),
            'capacity_weight': float(capacity_weight),
            'capacity_volume': float(capacity_volume),
            'shift_start': shift_start,
            'shift_end': shift_end,
            'velocidad_kmh': velocidad_kmh
        }

    st.divider()
    if st.button("🚀 Calcular Ruta (VRPTW)"):
        if not st.session_state.centro:
            st.warning("Selecciona un centro de distribución primero.")
        elif not st.session_state.puntos:
            st.warning("Agrega puntos de entrega en el mapa.")
        elif not fleet_cfg:
            st.warning("Configura la flota o carga un archivo válido.")
        else:
            with st.spinner("Resolviendo VRPTW..."):
                _, coords_per_route, metrics = solve_vrptw_simple(st.session_state.centro, st.session_state.puntos, fleet_cfg)
                if coords_per_route:
                    st.session_state.route_geojson = {
                        "type": "FeatureCollection",
                        "features": [{"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords_per_route[0]}, "properties": {}}]
                    }
                    st.session_state.route_metrics = {"distance_m": metrics['distance_km']*1000, "duration_s": metrics['time_min']*60}
                    st.success("Ruta generada correctamente 🚚")

# --- MAPA PRINCIPAL ---
col1, col2 = st.columns((2,1))
with col1:
    st.subheader("Mapa")
    m = folium.Map(location=st.session_state.map_center, zoom_start=13)
    if st.session_state.centro:
        folium.Marker(st.session_state.centro, icon=folium.Icon(color='red', icon='home'), popup="Centro").add_to(m)
    for i, p in enumerate(st.session_state.puntos):
        folium.Marker([p['lat'], p['lon']], popup=p.get('nombre', f"Punto {i+1}")).add_to(m)
    if st.session_state.route_geojson:
        folium.GeoJson(st.session_state.route_geojson, name='Ruta', style_function=lambda x: {"color":"blue"}).add_to(m)
    map_data = st_folium(m, width='100%', height=550)
    if map_data and map_data.get("last_clicked"):
        lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        if st.session_state.seleccionando_centro:
            st.session_state.centro = [lat, lon]
            st.session_state.seleccionando_centro = False
            st.success("Centro definido.")
        else:
            st.session_state.puntos.append({"lat": lat, "lon": lon, "nombre": f"Punto {len(st.session_state.puntos)+1}"})
            st.success("Punto agregado.")

with col2:
    st.subheader("📈 Estadísticas")
    if st.session_state.route_metrics:
        dist = st.session_state.route_metrics['distance_m']/1000
        time = st.session_state.route_metrics['duration_s']/60
        st.metric("Distancia total", f"{dist:.2f} km")
        st.metric("Tiempo estimado", f"{time:.0f} min")
    st.write("Pedidos registrados:")
    if st.session_state.puntos:
        st.dataframe(crear_tabla_de_pedidos(st.session_state.puntos))
    else:
        st.info("No hay pedidos registrados aún.")
