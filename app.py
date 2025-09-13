import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Panel de Control de Rutas",
    page_icon="🗺️",
    layout="wide"
)

# --- INICIALIZACIÓN DEL ESTADO ---
if 'puntos' not in st.session_state:
    st.session_state.puntos = []
if 'map_center' not in st.session_state:
    st.session_state.map_center = [4.60971, -74.08175]  # Bogotá
if 'centro' not in st.session_state:
    st.session_state.centro = None
if 'seleccionando_centro' not in st.session_state:
    st.session_state.seleccionando_centro = False

# --- FUNCIONES DE LÓGICA ---

def get_coords_from_city(city_name):
    """Obtiene las coordenadas (lat, lon) de una ciudad usando geopy."""
    try:
        geolocator = Nominatim(user_agent="routing_app")
        location = geolocator.geocode(city_name)
        if location:
            return [location.latitude, location.longitude]
    except Exception as e:
        st.error(f"No se pudo encontrar la ciudad. Error: {e}")
    return None

def simular_metricas_ruta(num_puntos):
    """Simula las métricas de la ruta."""
    if num_puntos == 0:
        return {"distancia": "0.00 km", "tiempo": "0 min"}
    
    distancia_base = 5
    tiempo_por_punto = 8
    distancia_total = distancia_base + num_puntos * np.random.uniform(1.5, 3.5)
    tiempo_estimado = num_puntos * (tiempo_por_punto + np.random.uniform(-2, 2))
    
    return {
        "distancia": f"{distancia_total:.2f} km",
        "tiempo": f"{tiempo_estimado:.0f} min"
    }

def crear_tabla_de_pedidos(puntos):
    """Crea un DataFrame con los detalles de los pedidos."""
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

# --- INTERFAZ DE USUARIO ---

st.title("🗺️ Panel de Control para Optimización de Rutas")
st.write(
    "Herramienta interactiva para planificar rutas. **Haz clic en el mapa para agregar los puntos de entrega** "
    "y configúralos en la barra lateral."
)

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ Configuración de la Ruta")
    
    # Ubicar mapa en ciudad
    ciudad = st.text_input("📍 Tu ciudad de operación", "Bogotá")
    if st.button("Buscar Ciudad"):
        new_coords = get_coords_from_city(ciudad)
        if new_coords:
            st.session_state.map_center = new_coords
            st.success(f"Mapa centrado en {ciudad}.")

    st.divider()

    # Punto central
    st.subheader("Centro de Distribución")
    if st.button("📍 Seleccionar Centro en el Mapa"):
        st.session_state.seleccionando_centro = True
        st.info("Haz clic en el mapa para definir el centro.")

    st.divider()

    # Pedidos
    st.subheader("Pedidos y Prioridades")
    st.caption("Los puntos agregados en el mapa aparecerán aquí.")

    for i, punto in enumerate(st.session_state.puntos):
        st.markdown(f"**Punto de Entrega {i+1}**")
        punto['nombre'] = st.text_input(f"Nombre del Destino", 
                                        value=punto.get('nombre', f'Punto {i+1}'), 
                                        key=f"nombre_{i}")
        punto['prioridad'] = st.selectbox(
            "Prioridad",
            ('Baja', 'Media', 'Alta', 'Muy Alta'),
            index=1,
            key=f"prioridad_{i}"
        )

    if st.button("🗑️ Limpiar todos los puntos"):
        st.session_state.puntos = []
        st.rerun()

# --- CONTENIDO PRINCIPAL ---
col1, col2 = st.columns((2, 1))

with col1:
    st.subheader("Mapa de la Ruta")
    m = folium.Map(location=st.session_state.map_center, zoom_start=13)

    # Marcar centro
    if st.session_state.centro:
        folium.Marker(
            st.session_state.centro,
            icon=folium.Icon(color="red", icon="home"),
            popup="Centro de Distribución"
        ).add_to(m)

    # Marcar pedidos
    for i, punto in enumerate(st.session_state.puntos):
        folium.Marker(
            [punto['lat'], punto['lon']], 
            popup=f"Destino: {punto.get('nombre', i+1)}\nPrioridad: {punto.get('prioridad', 'Media')}",
            tooltip=f"Punto {i+1}"
        ).add_to(m)

    # Dibujar rutas como líneas rectas
    if st.session_state.centro:
        for punto in st.session_state.puntos:
            folium.PolyLine(
                [st.session_state.centro, [punto['lat'], punto['lon']]],
                color="blue",
                weight=2,
                opacity=0.7
            ).add_to(m)

    # Capturar clics
    map_data = st_folium(m, width='100%', height=500)

    if map_data and map_data.get("last_clicked"):
        lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        if st.session_state.seleccionando_centro:
            st.session_state.centro = [lat, lon]
            st.session_state.seleccionando_centro = False
            st.rerun()
        else:
            st.session_state.puntos.append({"lat": lat, "lon": lon})
            st.rerun()

with col2:
    st.subheader("Estadísticas Clave")
    num_pedidos = len(st.session_state.puntos)
    metricas = simular_metricas_ruta(num_pedidos)
    st.metric(label="Pedidos Totales", value=num_pedidos)
    st.metric(label="Distancia Total Estimada", value=metricas["distancia"])
    st.metric(label="Tiempo Estimado de Viaje", value=metricas["tiempo"])

st.subheader("📄 Detalles de los Pedidos")
df_pedidos = crear_tabla_de_pedidos(st.session_state.puntos)
st.dataframe(df_pedidos, use_container_width=True)
