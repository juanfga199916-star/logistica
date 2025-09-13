import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import openrouteservice
import hashlib
import json
from openrouteservice import convert

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

# Storage for computed route and its "fingerprint" (to know if recompute is needed)
if 'route_geojson' not in st.session_state:
    st.session_state.route_geojson = None
if 'route_metrics' not in st.session_state:
    st.session_state.route_metrics = None
if 'last_click' not in st.session_state:
    st.session_state.last_click = None
if 'last_fingerprint' not in st.session_state:
    st.session_state.last_fingerprint = None

# --- CONFIGURAR ORS ---
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImY5MTA5MmE2NzVmZDRhYjBhMTk4YjZiNWNiMWY2YjQzIiwiaCI6Im11cm11cjY0In0="
client = openrouteservice.Client(key=ORS_API_KEY)

# --- HELPERS ---
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
        return pd.DataFrame(columns=['ID Pedido', 'Destino', 'Prioridad', 'Latitud', 'Longitud'])
    data = {
        'ID Pedido': [f"Pedido {i+1}" for i in range(len(puntos))],
        'Destino': [p.get('nombre', f'Punto {i+1}') for i, p in enumerate(puntos)],
        'Prioridad': [p.get('prioridad', 'Media') for p in puntos],
        'Latitud': [p['lat'] for p in puntos],
        'Longitud': [p['lon'] for p in puntos]
    }
    return pd.DataFrame(data)

def fingerprint(centro, puntos):
    """Crea un hash simple para detectar cambios en centro/puntos/prioridades."""
    obj = {
        'centro': centro,
        'puntos': [(p['lat'], p['lon'], p.get('prioridad','Media')) for p in puntos]
    }
    s = json.dumps(obj, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()

def calcular_ruta_ors(centro, puntos):
    """Llamada a ORS. Devuelve geojson y métricas (distancia m, duración s)."""
    if not centro or not puntos:
        return None, None

    # Ordenar por prioridad
    prioridad_ranking = {"Muy Alta": 1, "Alta": 2, "Media": 3, "Baja": 4}
    puntos_ordenados = sorted(puntos, key=lambda p: prioridad_ranking.get(p.get('prioridad','Media')))

    # Limitar número de puntos para evitar timeouts (puedes ajustar)
    MAX_POINTS = 50
    if len(puntos_ordenados) > MAX_POINTS:
        st.warning(f"Has agregado {len(puntos_ordenados)} puntos. Se usará solo los primeros {MAX_POINTS} por rendimiento.")
        puntos_ordenados = puntos_ordenados[:MAX_POINTS]

    coords = [[centro[1], centro[0]]] + [[p['lon'], p['lat']] for p in puntos_ordenados]

    try:
        route = client.directions(coords, profile='driving-car', format='geojson', optimize=False)
        feat = route.get('features', [None])[0]
        if not feat:
            return None, None
        segments = feat.get('properties', {}).get('segments', [])
        # segments can contain one object for whole route; sum distances/durations
        total_distance = sum(seg.get('distance', 0) for seg in segments)
        total_duration = sum(seg.get('duration', 0) for seg in segments)
        return route, {"distance_m": total_distance, "duration_s": total_duration}
    except openrouteservice.exceptions.ApiError as e:
        st.error(f"ORS API Error: {e}")
    except Exception as e:
        st.error(f"Error calculando ruta: {e}")
    return None, None

# --- UI ---
st.title("🗺️ Panel de Control para Optimización de Rutas")
st.write("Haz clic en el mapa para agregar pedidos. Define centro, prioridades y presiona 'Calcular Ruta'.")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuración")
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
    st.subheader("Pedidos")
    st.caption("Define nombre y prioridad de cada pedido.")

    # Editable fields for each punto (keys ensure stable widgets)
    for i, punto in enumerate(st.session_state.puntos):
        st.markdown(f"**Punto {i+1}**")
        punto['nombre'] = st.text_input(f"Nombre {i+1}", value=punto.get('nombre', f'Punto {i+1}'), key=f"nombre_{i}")
        punto['prioridad'] = st.selectbox("Prioridad", ('Baja','Media','Alta','Muy Alta'), index=['Baja','Media','Alta','Muy Alta'].index(punto.get('prioridad','Media')), key=f"prio_{i}")

    if st.button("🗑️ Limpiar todos los puntos"):
        st.session_state.puntos = []
        st.session_state.route_geojson = None
        st.session_state.route_metrics = None

    st.write("")  # espacio
    # Button to compute route once
    if st.button("🚀 Calcular Ruta (usar ORS)"):
        # compute only if centro + puntos exist
        if not st.session_state.centro:
            st.warning("Define primero el centro de distribución.")
        elif not st.session_state.puntos:
            st.warning("Agrega al menos un pedido en el mapa.")
        else:
            # fingerprint to avoid redundant recalculations
            fp = fingerprint(st.session_state.centro, st.session_state.puntos)
            if fp == st.session_state.last_fingerprint and st.session_state.route_geojson:
                st.info("La ruta ya está calculada y actualizada.")
            else:
                with st.spinner("Calculando ruta por calles (ORS)..."):
                    route, metrics = calcular_ruta_ors(st.session_state.centro, st.session_state.puntos)
                    if route:
                        st.session_state.route_geojson = route
                        st.session_state.route_metrics = metrics
                        st.session_state.last_fingerprint = fp
                        st.success("Ruta calculada y guardada.")
                    else:
                        st.error("No se pudo calcular la ruta. Revisa la clave ORS o la conectividad.")

# Main columns
col1, col2 = st.columns((2,1))

with col1:
    st.subheader("Mapa")
    m = folium.Map(location=st.session_state.map_center, zoom_start=13)

    # Draw center
    if st.session_state.centro:
        folium.Marker(st.session_state.centro, icon=folium.Icon(color='red', icon='home'), popup="Centro").add_to(m)

    # Draw points
    for i, punto in enumerate(st.session_state.puntos):
        folium.Marker([punto['lat'], punto['lon']], popup=f"{punto.get('nombre','')}\n{punto.get('prioridad','')}", tooltip=f"Punto {i+1}").add_to(m)

    # If route geojson exists, add it (this avoids recomputing on each rerun)
    if st.session_state.route_geojson:
        folium.GeoJson(st.session_state.route_geojson, name='Ruta', style_function=lambda x: {"color":"blue","weight":4,"opacity":0.8}).add_to(m)
    else:
        # If no route, optionally draw straight lines from center to points as visual guide
        if st.session_state.centro:
            for punto in st.session_state.puntos:
                folium.PolyLine([st.session_state.centro, [punto['lat'], punto['lon']]], color='gray', weight=1, dash_array='5').add_to(m)

    # Render map and capture clicks
    map_data = st_folium(m, width='100%', height=600)

    # Handle click: use last_click caching to avoid duplicates
    if map_data and map_data.get("last_clicked"):
        clicked = map_data["last_clicked"]
        lat, lon = clicked["lat"], clicked["lng"]
        # ignore if identical to previous click (same lat/lon)
        if st.session_state.last_click and abs(st.session_state.last_click[0]-lat) < 1e-9 and abs(st.session_state.last_click[1]-lon) < 1e-9:
            pass
        else:
            st.session_state.last_click = (lat, lon)
            if st.session_state.seleccionando_centro:
                st.session_state.centro = [lat, lon]
                st.session_state.seleccionando_centro = False
                # clear previous route because center changed
                st.session_state.route_geojson = None
                st.session_state.route_metrics = None
                st.success("Centro definido.")
            else:
                # limit max points to prevent ORS overload
                MAX_POINTS = 100
                if len(st.session_state.puntos) >= MAX_POINTS:
                    st.warning(f"Has alcanzado el límite máximo de {MAX_POINTS} puntos.")
                else:
                    st.session_state.puntos.append({"lat": lat, "lon": lon, "prioridad": "Media", "nombre": f"Punto {len(st.session_state.puntos)+1}"})
                    # clear previous route because points changed
                    st.session_state.route_geojson = None
                    st.session_state.route_metrics = None
                    st.success(f"Punto {len(st.session_state.puntos)} agregado.")

with col2:
    st.subheader("Estadísticas")
    num = len(st.session_state.puntos)
    st.metric("Pedidos Totales", num)
    # Prefer metrics from ORS if available
    if st.session_state.route_metrics:
        dist_km = st.session_state.route_metrics['distance_m'] / 1000
        dur_min = st.session_state.route_metrics['duration_s'] / 60
        st.metric("Distancia (real)", f"{dist_km:.2f} km")
        st.metric("Tiempo (real)", f"{dur_min:.0f} min")
    else:
        # fallback to simulated
        sim = lambda n: (5 + n * np.random.uniform(1.5, 3.5), n*(8 + np.random.uniform(-2,2)))
        d, t = sim(num)
        st.metric("Distancia (estimada)", f"{d:.2f} km")
        st.metric("Tiempo (estimado)", f"{t:.0f} min")

st.subheader("📄 Detalles de Pedidos")
df = crear_tabla_de_pedidos(st.session_state.puntos)
st.dataframe(df, use_container_width=True)
