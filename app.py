# ============================================================
#  APP DE HELADAS CON PREDICCIÓN REAL
# ============================================================

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from datetime import datetime

# Configuración
st.set_page_config(page_title="Heladas Madrid", page_icon="snowflake", layout="wide")
st.title("Sistema de Alerta de Heladas - Madrid, Cundinamarca")

# ============================================================
# IMPORTAR PREDICTOR
# ============================================================
try:
    from predictor import PredictorHeladas
    PREDICTOR_DISPONIBLE = True
except Exception as e:
    st.error(f"No se pudo importar el predictor: {e}")
    PREDICTOR_DISPONIBLE = False

# ============================================================
# CARGAR PREDICTOR
# ============================================================
@st.cache_resource
def cargar_predictor():
    try:
        return PredictorHeladas()
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Configuración")
if st.sidebar.button("Actualizar Predicción", type="primary"):
    st.cache_resource.clear()
    st.rerun()

# ============================================================
# HACER PREDICCIÓN
# ============================================================
if not PREDICTOR_DISPONIBLE:
    st.warning("Predictor no disponible. Usando valores por defecto.")
    temp_predicha = 1.5
    prob_helada = 65
    riesgo = "MEDIO"
    color_riesgo = "yellow"
    color_mapa = "orange"
    resultado = None
else:
    predictor = cargar_predictor()
    if predictor is None:
        temp_predicha = 1.5
        prob_helada = 65
        riesgo = "MEDIO"
        color_riesgo = "yellow"
        color_mapa = "orange"
        resultado = None
    else:
        with st.spinner("Generando predicción para mañana..."):
            resultado = predictor.predecir()

        if "error" in resultado:
            st.error(f"Error en predicción: {resultado['error']}")
            temp_predicha = 1.5
            prob_helada = 65
            riesgo = "MEDIO"
            color_riesgo = "yellow"
            color_mapa = "orange"
        else:
            temp_predicha = resultado['temperatura_predicha']
            prob_helada = resultado['probabilidad_helada']
            riesgo = resultado['riesgo']
            color_riesgo = resultado['emoji_riesgo']
            color_mapa = resultado['color_mapa']

            # Debug en sidebar
            st.sidebar.subheader("Información de Predicción")
            st.sidebar.write(f"Fecha consulta: {resultado['fecha_consulta']}")
            st.sidebar.write(f"Predicción para: **{resultado['fecha_prediccion']}**")
            st.sidebar.write(f"Temp. ayer: {resultado['temp_ayer']:.1f}°C")
            st.sidebar.write(f"Cambio esperado: {resultado['cambio_esperado']:+.1f}°C")
            st.sidebar.write(f"Promedio 7 días: {resultado['temp_promedio_7d']:.1f}°C")
            st.sidebar.write(f"Mínima 7 días: {resultado['temp_minima_7d']:.1f}°C")
            st.sidebar.write(f"Máxima 7 días: {resultado['temp_maxima_7d']:.1f}°C")
            st.success("Predicción actualizada correctamente")

# ============================================================
# MÉTRICAS PRINCIPALES
# ============================================================
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Temperatura Predicha (mañana)", f"{temp_predicha:.1f}°C",
              delta=f"{resultado['cambio_esperado']:+.1f}°C" if resultado else None)
with col2:
    st.metric("Probabilidad Helada", f"{prob_helada:.1f}%")
with col3:
    st.metric("Nivel de Riesgo", f"{color_riesgo} {riesgo}")

# ============================================================
# ALERTA VISUAL
# ============================================================
st.markdown("---")
if temp_predicha <= 0:
    st.error("**ALERTA DE HELADA**: Se espera temperatura bajo 0°C mañana")
elif temp_predicha <= 2:
    st.warning("**PRECAUCIÓN**: Temperatura muy cercana al punto de congelación")
else:
    st.success("**SIN RIESGO**: No se espera helada para mañana")

# ============================================================
# MAPA
# ============================================================
st.subheader("Mapa de Temperatura - Madrid, Cundinamarca")
madrid_lat, madrid_lon = 4.7333, -74.2667
mapa = folium.Map(location=[madrid_lat, madrid_lon], zoom_start=13, tiles='OpenStreetMap')

folium.Marker(
    location=[madrid_lat, madrid_lon],
    popup=f"<b>Madrid, Cundinamarca</b><br>Predicción mañana: {temp_predicha:.1f}°C<br>Riesgo: {riesgo}",
    tooltip=f"{temp_predicha:.1f}°C - {riesgo}",
    icon=folium.Icon(color=color_mapa, icon='thermometer-half', prefix='fa')
).add_to(mapa)

folium.Circle(
    location=[madrid_lat, madrid_lon],
    radius=2500,
    color=color_mapa,
    fill=True,
    fillOpacity=0.2,
    popup="Zona de influencia"
).add_to(mapa)

st_folium(mapa, width=700, height=500)

# ============================================================
# HISTORIAL Y ESTADÍSTICAS
# ============================================================
if resultado and PREDICTOR_DISPONIBLE and predictor:
    st.markdown("---")
    st.subheader("Historial de Temperatura (Últimos 30 días)")
    historial = resultado['historial_30d']
    st.line_chart(historial.set_index('Fecha')[predictor.target], use_container_width=True)

    with st.expander("Ver Estadísticas Generales"):
        stats = predictor.estadisticas_generales()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Registros", stats['total_registros'])
        c2.metric("Temp. Promedio", f"{stats['temp_promedio']:.1f}°C")
        c3.metric("Heladas Totales", stats['heladas_totales'])
        c4.metric("% Heladas", f"{stats['porcentaje_heladas']:.1f}%")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.info("Este sistema utiliza modelos de Machine Learning entrenados con datos históricos del IDEAM para predecir heladas en Madrid, Cundinamarca.")
st.caption(f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Presiona 'Actualizar Predicción' para refrescar")
