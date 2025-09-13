import pytest
import pandas as pd
from app import crear_tabla_de_pedidos, simular_metricas_ruta

# --- Prueba para la nueva lógica de la tabla de pedidos ---
def test_crear_tabla_de_pedidos_con_datos():
    """
    Verifica que la tabla se cree correctamente a partir de una lista de puntos.
    """
    # 1. PREPARACIÓN: Creamos una lista de puntos similar a la de st.session_state
    puntos_de_prueba = [
        {"lat": 4.6, "lon": -74.0, "nombre": "Cliente A", "prioridad": "Alta"},
        {"lat": 4.7, "lon": -74.1, "nombre": "Cliente B", "prioridad": "Baja"}
    ]

    # 2. ACCIÓN
    df_resultado = crear_tabla_de_pedidos(puntos_de_prueba)

    # 3. VERIFICACIÓN
    assert isinstance(df_resultado, pd.DataFrame)
    assert len(df_resultado) == 2
    assert df_resultado.iloc[0]['Destino'] == "Cliente A"
    assert df_resultado.iloc[1]['Prioridad'] == "Baja"
    assert 'Latitud' in df_resultado.columns

def test_crear_tabla_de_pedidos_vacia():
    """
    Verifica que la función devuelva un DataFrame vacío si no hay puntos.
    """
    # 1. PREPARACIÓN
    puntos_vacios = []

    # 2. ACCIÓN
    df_resultado = crear_tabla_de_pedidos(puntos_vacios)

    # 3. VERIFICACIÓN
    assert isinstance(df_resultado, pd.DataFrame)
    assert len(df_resultado) == 0

def test_simular_metricas_ruta_cero_puntos():
    """
    Verifica que las métricas sean cero si no hay puntos.
    """
    metricas = simular_metricas_ruta(0)
    assert metricas["distancia"] == "0.00 km"
    assert metricas["tiempo"] == "0 min"
