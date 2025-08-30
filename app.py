import streamlit as st
import pandas as pd
import numpy as np

# --- Configuración de la Página ---
# st.set_page_config se debe llamar al principio del script.
st.set_page_config(
    page_title="Mi Primera App de Rutas",
    page_icon="🗺️",
    layout="wide"
)

# --- Título y Descripción ---
st.title("🗺️ Panel de Control para Optimización de Rutas")
st.write(
    "¡Bienvenido! Esta es una herramienta interactiva para visualizar y planificar "
    "rutas de manera eficiente. Usa los controles en la barra lateral para comenzar."
)

# --- Barra Lateral (Sidebar) ---
st.sidebar.header("Configuración de la Ruta")

# Widgets para introducir datos en la barra lateral
punto_inicio = st.sidebar.text_input("📍 Punto de Inicio", "Almacén Central")
puntos_entrega = st.sidebar.slider(
    "🚚 ¿Cuántos pedidos tienes?",
    min_value=1,
    max_value=10,
    value=5, # Valor inicial
    step=1
)

# Botón para ejecutar la simulación/cálculo
if st.sidebar.button("Calcular Ruta Óptima"):
    # Esta es una acción que se ejecuta cuando se presiona el botón
    st.sidebar.success(f"Calculando la mejor ruta para {puntos_entrega} pedidos desde {punto_inicio}.")
    
    # --- Contenido Principal ---
    
    # Usamos columnas para organizar el contenido
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Mapa de la Ruta (Simulado)")
        
        # Crear datos de ejemplo para el mapa
        # En un caso real, aquí irían las coordenadas de la ruta calculada
        map_data = pd.DataFrame(
            np.random.randn(puntos_entrega, 2) / [20, 20] + [4.60971, -74.08175], # Coordenadas base de Bogotá
            columns=['lat', 'lon']
        )
        st.map(map_data)
        st.caption("Visualización de los puntos de entrega.")

    with col2:
        st.subheader("Estadísticas de la Ruta")
        
        # Mostramos algunas métricas como en tu imagen de ejemplo
        distancia_total = np.random.uniform(5, 25)
        tiempo_estimado = distancia_total * np.random.uniform(3, 8) # en minutos

        st.metric(label="Pedidos Totales", value=f"{puntos_entrega}")
        st.metric(label="Distancia Total Estimada", value=f"{distancia_total:.2f} km")
        st.metric(label="Tiempo Estimado de Viaje", value=f"{tiempo_estimado:.0f} min")
        
        st.info("Estas son métricas simuladas. La lógica de cálculo real se implementará aquí.")

    st.subheader("Detalles de los Pedidos")
    # Crear un DataFrame de ejemplo para mostrar una tabla
    df_pedidos = pd.DataFrame({
        'Pedido ID': range(1, puntos_entrega + 1),
        'Destino': [f'Punto {i}' for i in range(1, puntos_entrega + 1)],
        'Prioridad': np.random.choice(['Alta', 'Media', 'Baja'], puntos_entrega)
    })
    st.dataframe(df_pedidos)

else:
    # Mensaje que se muestra antes de presionar el botón
    st.info("Ajusta los parámetros en la barra lateral y haz clic en 'Calcular Ruta Óptima' para empezar.")
