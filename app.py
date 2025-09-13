import streamlit as st
import pandas as pd
import numpy as np
from streamlit_folium import st_folium
import folium

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Panel de Control de Rutas",
    page_icon="🗺️",
    layout="wide"
)

# --- INICIALIZACIÓN DEL ESTADO DE LA SESIÓN ---
# 'session_state' es la memoria de Streamlit. Lo usamos para guardar los puntos
# que el usuario agrega en el mapa, para que no se borren.
if 'puntos' not in st.session_state:
    st.session_state.puntos = []

# --- FUNCIONES DE LÓGICA ---

def simular_metricas_ruta(num_puntos):
    """Simula métricas de la ruta basadas en el número de entregas."""
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
    """Crea un DataFrame con los detalles de los puntos de entrega seleccionados."""
    if not puntos:
        return pd.DataFrame(columns=['ID Pedido', 'Destino', 'Prioridad', 'Latitud', 'Longitud'])
    
    # Extraemos los datos de la lista de diccionarios en el session_state
    ids = [f"Pedido {i+1}" for i in range(len(puntos))]
    destinos = [p.get('nombre', f'Punto {i+1}') for i, p in enumerate(puntos)]
    prioridades = [p.get('prioridad', 'Media') for p in puntos]
    latitudes = [p['lat'] for p in puntos]
    longitudes = [p['lon'] for p in puntos]

    df_pedidos = pd.DataFrame({
        'ID Pedido': ids,
        'Destino': destinos,
        'Prioridad': prioridades,
        'Latitud': latitudes,
        'Longitud': longitudes
    })
    return df_pedidos

# --- INTERFAZ DE USUARIO ---

st.title("🗺️ Panel de Control para Optimización de Rutas")
st.write(
    "Herramienta interactiva para planificar rutas. **Haz clic en el mapa para agregar los puntos de entrega** y configúralos en la barra lateral."
)

# --- BARRA LATERAL (CONTROLES) ---
with st.sidebar:
    st.header("⚙️ Configuración de la Ruta")
    punto_inicio = st.text_input("📍 Punto de Inicio", "Almacén Central")

    st.subheader("Pedidos y Prioridades")
    st.caption("Los puntos agregados en el mapa aparecerán aquí.")

    # Iteramos sobre cada punto guardado en el estado para mostrar sus opciones
    for i, punto in enumerate(st.session_state.puntos):
        st.markdown(f"**Punto {i+1}**")
        # Creamos campos para que el usuario asigne nombre y prioridad a cada punto
        punto['nombre'] = st.text_input(f"Nombre del Destino {i+1}", value=punto.get('nombre', f'Punto {i+1}'), key=f"nombre_{i}")
        punto['prioridad'] = st.selectbox("Prioridad", ['Alta', 'Media', 'Baja'], index=1, key=f"prioridad_{i}")

    # Botón para limpiar todos los puntos agregados
    if st.button("🗑️ Limpiar todos los puntos"):
        st.session_state.puntos = []
        st.rerun() # Volvemos a ejecutar el script para refrescar la UI

# --- MAPA INTERACTIVO Y LÓGICA PRINCIPAL ---
col1, col2 = st.columns((2, 1))

with col1:
    st.subheader("Selecciona los Puntos de Entrega")
    
    # Creamos un mapa centrado en Bogotá
    m = folium.Map(location=[4.60971, -74.08175], zoom_start=12)

    # Añadimos los marcadores de los puntos ya existentes al mapa
    for i, punto in enumerate(st.session_state.puntos):
        folium.Marker(
            [punto['lat'], punto['lon']], 
            popup=f"Punto {i+1}",
            tooltip=f"Destino: {punto.get('nombre', i+1)}"
        ).add_to(m)

    # Usamos st_folium para renderizar el mapa y capturar clics
    map_data = st_folium(m, width='100%')

    # Si el usuario hace clic en el mapa, 'map_data' contendrá la información
    if map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]
        
        # Guardamos el nuevo punto en nuestra lista de 'session_state'
        st.session_state.puntos.append({"lat": lat, "lon": lon})
        # Forzamos un 'rerun' para que la barra lateral se actualice instantáneamente
        st.rerun()

with col2:
    st.subheader("Estadísticas Clave")
    
    num_pedidos = len(st.session_state.puntos)
    metricas = simular_metricas_ruta(num_pedidos)
    
    st.metric(label="Pedidos Totales", value=num_pedidos)
    st.metric(label="Distancia Total Estimada", value=metricas["distancia"])
    st.metric(label="Tiempo Estimado de Viaje", value=metricas["tiempo"])

st.subheader("📄 Detalles de los Pedidos")
# Creamos y mostramos la tabla de pedidos a partir de los puntos seleccionados
df_pedidos = crear_tabla_de_pedidos(st.session_state.puntos)
st.dataframe(df_pedidos)
