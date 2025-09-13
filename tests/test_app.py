import pytest
import pandas as pd
from app import generar_puntos_de_entrega # Importamos la función a probar

# --- Caso 1: Prueba con una entrada válida ---
def test_generar_puntos_de_entrega_caso_valido():
    """
    Verifica que la función genere correctamente un DataFrame con un número
    positivo de puntos.
    """
    # 1. PREPARACIÓN (Arrange)
    num_puntos = 5
    
    # 2. ACCIÓN (Act)
    resultado = generar_puntos_de_entrega(num_puntos)
    
    # 3. VERIFICACIÓN (Assert)
    # Comprueba que el resultado es un DataFrame de pandas
    assert isinstance(resultado, pd.DataFrame), "El resultado debe ser un DataFrame de pandas."
    
    # Comprueba que el DataFrame tiene el número correcto de filas
    assert len(resultado) == num_puntos, f"El DataFrame debería tener {num_puntos} filas."
    
    # Comprueba que las columnas necesarias ('lat', 'lon') existen
    assert 'lat' in resultado.columns, "La columna 'lat' debe existir."
    assert 'lon' in resultado.columns, "La columna 'lon' debe existir."


# --- Caso 2: Prueba con una entrada que debe generar un error (caso límite) ---
def test_generar_puntos_de_entrega_caso_invalido_negativo():
    """
    Verifica que la función levante un error (ValueError) si se le pasa
    un número negativo, ya que no se pueden generar -1 puntos.
    """
    # 1. PREPARACIÓN (Arrange)
    num_puntos_invalidos = -1
    
    # 2. ACCIÓN y VERIFICACIÓN (Act & Assert)
    # Usamos pytest.raises para verificar que se produce un error específico.
    # El bloque de código dentro de 'with' debe levantar un ValueError para que
    # la prueba pase.
    with pytest.raises(ValueError):
        generar_puntos_de_entrega(num_puntos_invalidos)

# --- (Opcional) Otro caso límite muy útil ---
def test_generar_puntos_de_entrega_caso_cero():
    """
    Verifica que la función maneja correctamente el caso de 0 puntos,
    devolviendo un DataFrame vacío.
    """
    # 1. PREPARACIÓN (Arrange)
    num_puntos = 0
    
    # 2. ACCIÓN (Act)
    resultado = generar_puntos_de_entrega(num_puntos)
    
    # 3. VERIFICACIÓN (Assert)
    assert isinstance(resultado, pd.DataFrame), "El resultado debe ser un DataFrame."
    assert len(resultado) == num_puntos, "El DataFrame debería estar vacío (0 filas)."
