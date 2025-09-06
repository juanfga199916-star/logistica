# logistica
🗺️ Panel de Control para Optimización de RutasEste es un proyecto de Streamlit que proporciona una herramienta interactiva para simular y visualizar la planificación de rutas logísticas. Permite a los usuarios configurar el número de entregas y ver una representación visual en un mapa, junto con métricas clave de la operación.🚀 Cómo Ejecutar el Proyecto LocalmenteSigue estos pasos para poner en marcha la aplicación en tu máquina local.1. PrerrequisitosTener instalado Python 3.8 o superior.Tener git instalado (opcional, para clonar el repositorio).2. Configuración del EntornoClona el repositorio (o descarga los archivos):git clone <URL-de-tu-repositorio-en-github>
cd proyecto-logistica
Crea y activa un entorno virtual:En Windows:python -m venv venv
.\venv\Scripts\activate
En macOS / Linux:python3 -m venv venv
source venv/bin/activate
Instala las dependencias:El archivo requirements.txt contiene todas las librerías necesarias.pip install -r requirements.txt
3. Ejecuta la AplicaciónUna vez que las dependencias estén instaladas, puedes iniciar la aplicación Streamlit con el siguiente comando:streamlit run app.py
¡Tu navegador se abrirá automáticamente con la aplicación en funcionamiento!☁️ Despliegue en Streamlit Community CloudEste repositorio está listo para ser desplegado. Simplemente crea una cuenta en Streamlit Community Cloud, conecta tu cuenta de GitHub y selecciona este repositorio para el despliegue. Streamlit leerá automáticamente el archivo requirements.txt e iniciará la aplicación.
