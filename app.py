import streamlit as st
import pandas as pd
import numpy as np

# --- CONFIGURACIÓN DE LA PÁGINA ---
# Es una buena práctica definir la configuración al inicio.
st.set_page_config(
    page_title="Panel de Control de Rutas",
    page_icon="🚚",
    layout="wide"
)

# --- FUNCIONES DE LÓGICA Y DATOS ---
# Separar la lógica de la presentación hace el código más limpio.

def generar_puntos_de_entrega(num_puntos, lat_centro=4.60971, lon_centro=-74.08175):
    """
    Genera un DataFrame con coordenadas aleatorias para los puntos de entrega.

    Args:
        num_puntos (int): El número de puntos a generar.
        lat_centro (float): La latitud del punto central del mapa.
        lon_centro (float): La longitud del punto central del mapa.

    Returns:
        pd.DataFrame: Un DataFrame con las columnas 'lat' y 'lon'.
    """
    # Genera datos aleatorios distribuidos normalmente alrededor de un punto central.
    map_data = pd.DataFrame(
        np.random.randn(num_puntos, 2) / [50, 50] + [lat_centro, lon_centro],
        columns=['lat', 'lon']
    )
    return map_data

def simular_metricas_ruta(num_puntos):
    """
    Simula métricas de la ruta basadas en el número de entregas.

    Args:
        num_puntos (int): El número de puntos de entrega.

    Returns:
        dict: Un diccionario con la distancia y el tiempo estimados.
    """
    # Lógica de simulación simple: más puntos implican más distancia y tiempo.
    distancia_base = 5
    tiempo_por_punto = 8
    
    distancia_total = distancia_base + num_puntos * np.random.uniform(1.5, 3.5)
    tiempo_estimado = num_puntos * (tiempo_por_punto + np.random.uniform(-2, 2))
    
    return {
        "distancia": f"{distancia_total:.2f} km",
        "tiempo": f"{tiempo_estimado:.0f} min"
    }

def crear_tabla_de_pedidos(num_puntos):
    """
    Crea un DataFrame de ejemplo con detalles de los pedidos.

    Args:
        num_puntos (int): El número de pedidos a generar en la tabla.

    Returns:
        pd.DataFrame: Una tabla con información simulada de los pedidos.
    """
    df_pedidos = pd.DataFrame({
        'ID Pedido': range(1, num_puntos + 1),
        'Destino': [f'Punto de Entrega {i}' for i in range(1, num_puntos + 1)],
        'Prioridad': np.random.choice(['Alta', 'Media', 'Baja'], num_puntos),
        'Estado': np.random.choice(['Pendiente', 'En Ruta'], num_puntos)
    })
    return df_pedidos

# --- INTERFAZ DE USUARIO ---

def main():
    """
    Función principal que construye y ejecuta la aplicación Streamlit.
    """
    st.title("🗺️ Panel de Control para Optimización de Rutas")
    st.write(
        "Herramienta interactiva para visualizar y planificar rutas de manera eficiente. "
        "Usa los controles en la barra lateral para generar una nueva simulación."
    )

    # --- BARRA LATERAL (CONTROLES) ---
    st.sidebar.header("⚙️ Configuración de la Ruta")
    
    punto_inicio = st.sidebar.text_input("📍 Punto de Inicio", "Almacén Central")
    puntos_entrega_slider = st.sidebar.slider(
        "🚚 Número de Entregas",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="Selecciona cuántos puntos de entrega deseas simular."
    )

    # --- LÓGICA DE EJECUCIÓN ---
    if st.sidebar.button("🚀 Calcular Ruta Óptima"):
        
        with st.spinner('Calculando la mejor ruta...'):
            # 1. Llamar a las funciones de lógica para obtener los datos
            map_data = generar_puntos_de_entrega(puntos_entrega_slider)
            metricas = simular_metricas_ruta(puntos_entrega_slider)
            df_pedidos = crear_tabla_de_pedidos(puntos_entrega_slider)

        st.success(f"¡Simulación completada para {puntos_entrega_slider} entregas desde {punto_inicio}!")

        # --- CONTENIDO PRINCIPAL (RESULTADOS) ---
        col1, col2 = st.columns((2, 1)) # Columna del mapa más ancha

        with col1:
            st.subheader("Mapa de la Ruta (Simulado)")
            st.map(map_data)
            st.caption("Visualización geográfica de los puntos de entrega.")

        with col2:
            st.subheader("Estadísticas Clave")
            st.metric(label="Pedidos Totales", value=puntos_entrega_slider)
            st.metric(label="Distancia Total Estimada", value=metricas["distancia"])
            st.metric(label="Tiempo Estimado de Viaje", value=metricas["tiempo"])
        
        st.subheader("📄 Detalles de los Pedidos")
        st.dataframe(df_pedidos)

    else:
        # Mensaje que se muestra antes de presionar el botón
        st.info("Ajusta los parámetros en la barra lateral y haz clic en 'Calcular Ruta Óptima' para empezar.")

# --- PUNTO DE ENTRADA ---
# Es una buena práctica en Python llamar a la función principal de esta manera.
if __name__ == "__main__":
    main()
