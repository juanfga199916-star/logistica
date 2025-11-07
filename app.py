
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

# --- CONFIGURAR ORS ---
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImY5MTA5MmE2NzVmZDRhYjBhMTk4YjZiNWNiMWY2YjQzIiwiaCI6Im11cm11cjY0In0="  # reemplaza si tienes una
client = openrouteservice.Client(key=ORS_API_KEY) if ORS_API_KEY else None

# --- FUNCIONES ---
def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia Haversine (km)."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    return 6371 * 2 * asin(sqrt(sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2))

def get_coords_from_city(city_name):
    """Geocodifica ciudad."""
    try:
        geolocator = Nominatim(user_agent="routing_app")
        location = geolocator.geocode(city_name, timeout=10)
        if location:
            return [location.latitude, location.longitude]
    except Exception as e:
        st.warning(f"No se pudo geocodificar la ciudad: {e}")
    return None

def crear_tabla_de_pedidos(puntos):
    """Convierte lista de puntos en DataFrame."""
    if not puntos:
        return pd.DataFrame(columns=['ID Pedido','Destino','Prioridad','Peso (kg)','Volumen (m³)','Service (min)','TW Inicio','TW Fin','Latitud','Longitud'])
    data = {
        'ID Pedido': [f"Pedido {i+1}" for i in range(len(puntos))],
        'Destino': [p.get('nombre', f"Punto {i+1}") for i,p in enumerate(puntos)],
        'Prioridad': [p.get('prioridad','Media') for p in puntos],
        'Peso (kg)': [p.get('peso',0.0) for p in puntos],
        'Volumen (m³)': [p.get('volumen',0.0) for p in puntos],
        'Service (min)': [p.get('service_time',5) for p in puntos],
        'TW Inicio': [p.get('tw_start','00:00') for p in puntos],
        'TW Fin': [p.get('tw_end','23:59') for p in puntos],
        'Latitud': [p.get('lat','') for p in puntos],
        'Longitud': [p.get('lon','') for p in puntos]
    }
    return pd.DataFrame(data)

# --- UI ---
st.title("🗺️ Panel de Control de Rutas con Carga de Archivo Único")
st.write("Sube un único archivo Excel que contenga los datos de los pedidos (y opcionalmente coordenadas).")

with st.sidebar:
    st.header("📂 Cargar Archivo Excel")
    uploaded_file = st.file_uploader("Selecciona el archivo Excel", type=["xlsx"])

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            st.success("Archivo leído correctamente ✅")
            st.dataframe(df)

            # Normalizar nombres de columnas
            df.columns = [c.strip().lower() for c in df.columns]

            # Verificar columnas requeridas
            required_cols = ['volumen','prioridad','tw_start','tw_end','service_time']
            if not all(col in df.columns for col in required_cols):
                st.error(f"Faltan columnas requeridas. Se esperaban: {required_cols}")
            else:
                # Si hay coordenadas, crear puntos automáticamente
                if 'lat' in df.columns and 'lon' in df.columns:
                    st.session_state.puntos = []
                    for i, row in df.iterrows():
                        st.session_state.puntos.append({
                            'lat': row['lat'],
                            'lon': row['lon'],
                            'nombre': f"Punto {i+1}",
                            'prioridad': row.get('prioridad','Media'),
                            'peso': row.get('peso',0.0),
                            'volumen': row.get('volumen',0.0),
                            'service_time': row.get('service_time',5),
                            'tw_start': row.get('tw_start','08:00'),
                            'tw_end': row.get('tw_end','18:00')
                        })
                    st.success(f"{len(st.session_state.puntos)} puntos cargados y marcados en el mapa.")
                else:
                    st.info("No se encontraron coordenadas. Podrás agregarlas manualmente en el mapa.")
                    st.session_state.pedidos_df = df

        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")

    st.divider()
    if st.button("📍 Seleccionar Centro en el Mapa"):
        st.session_state.seleccionando_centro = True
        st.info("Haz clic en el mapa para establecer el centro.")

# --- MAPA PRINCIPAL ---
col1, col2 = st.columns((2,1))
with col1:
    st.subheader("🗺️ Mapa de Pedidos")
    m = folium.Map(location=st.session_state.map_center, zoom_start=13)

    # Mostrar centro
    if st.session_state.centro:
        folium.Marker(st.session_state.centro, icon=folium.Icon(color='red', icon='home'), popup="Centro").add_to(m)

    # Mostrar pedidos cargados
    for i, p in enumerate(st.session_state.puntos):
        folium.Marker(
            [p['lat'], p['lon']],
            popup=f"{p.get('nombre')} - {p.get('prioridad')}",
            tooltip=f"Punto {i+1}"
        ).add_to(m)

    # Capturar clics del usuario
    map_data = st_folium(m, width='100%', height=550)
    if map_data and map_data.get("last_clicked"):
        lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        if st.session_state.seleccionando_centro:
            st.session_state.centro = [lat, lon]
            st.session_state.seleccionando_centro = False
            st.success("Centro de distribución definido.")
        else:
            df_base = st.session_state.get('pedidos_df', pd.DataFrame())
            if not df_base.empty:
                index = len(st.session_state.puntos)
                if index < len(df_base):
                    fila = df_base.iloc[index]
                    nuevo_punto = {
                        'lat': lat, 'lon': lon,
                        'nombre': f"Punto {index+1}",
                        'prioridad': fila.get('prioridad','Media'),
                        'peso': fila.get('peso',0.0),
                        'volumen': fila.get('volumen',0.0),
                        'service_time': fila.get('service_time',5),
                        'tw_start': fila.get('tw_start','08:00'),
                        'tw_end': fila.get('tw_end','18:00')
                    }
                    st.session_state.puntos.append(nuevo_punto)
                    st.success(f"Punto {index+1} agregado desde Excel.")
            else:
                st.session_state.puntos.append({'lat': lat, 'lon': lon, 'nombre': f"Punto {len(st.session_state.puntos)+1}"})
                st.success("Punto agregado manualmente.")

with col2:
    st.subheader("📋 Detalle de Pedidos")
    df_pedidos = crear_tabla_de_pedidos(st.session_state.puntos)
    st.dataframe(df_pedidos, use_container_width=True)
