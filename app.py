# ============================================================
#  APP DE HELADAS - VERSIÓN SIMPLIFICADA
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import locale

# Configurar locale para español
try:
    locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_TIME, 'es_ES')
    except:
        pass  # Si no funciona, continuará en inglés

# Configuración
st.set_page_config(
    page_title="Heladas Madrid",
    page_icon="❄️",
    layout="wide"
)

# Título
st.title("❄️ Sistema de Alerta de Heladas - Madrid, Cundinamarca")

# ============================================================
# IMPORTAR PREDICTOR
# ============================================================
try:
    from predictor import PredictorHeladas
    PREDICTOR_DISPONIBLE = True
except Exception as e:
    st.error(f"⚠️ No se pudo importar el predictor: {e}")
    PREDICTOR_DISPONIBLE = False

# ============================================================
# CARGAR PREDICTOR
# ============================================================
@st.cache_resource
def cargar_predictor():
    """Carga el predictor una sola vez"""
    try:
        return PredictorHeladas()
    except Exception as e:
        st.error(f"❌ Error cargando modelos: {e}")
        return None

# ============================================================
# BOTÓN DE ACTUALIZACIÓN
# ============================================================
if st.button("🔄 Actualizar Predicción", type="primary"):
    st.cache_resource.clear()
    st.rerun()

st.markdown("---")

# ============================================================
# HACER PREDICCIÓN
# ============================================================
if not PREDICTOR_DISPONIBLE:
    st.warning("⚠️ Predictor no disponible. Usando valores por defecto.")
    temp_predicha = 1.5
    prob_helada = 65
    riesgo = "MEDIO"
    color_riesgo = "🟡"
    color_mapa = "orange"
    resultado = None
    predicciones_7dias = []
else:
    predictor = cargar_predictor()
    
    if predictor is None:
        st.error("⚠️ No se pudo cargar el predictor. Usando valores por defecto.")
        temp_predicha = 1.5
        prob_helada = 65
        riesgo = "MEDIO"
        color_riesgo = "🟡"
        color_mapa = "orange"
        resultado = None
    else:
        # Hacer predicción para MAÑANA (usando fecha actual del sistema)
        with st.spinner("🔮 Generando predicción..."):
            resultado = predictor.predecir()
        
        if "error" in resultado:
            st.error(f"❌ Error en predicción: {resultado['error']}")
            temp_predicha = 1.5
            prob_helada = 65
            riesgo = "MEDIO"
            color_riesgo = "🟡"
            color_mapa = "orange"
        else:
            # Extraer resultados del PRIMER DÍA (mañana)
            temp_predicha = resultado['temperatura_predicha']
            prob_helada = resultado['probabilidad_helada']
            riesgo = resultado['riesgo']
            color_riesgo = resultado['emoji_riesgo']
            color_mapa = resultado['color_mapa']

# ============================================================
# MÉTRICAS PRINCIPALES (SOLO MAÑANA)
# ============================================================
if resultado:
    st.subheader(f"🌡️ Predicción para Mañana ({resultado['fecha_prediccion'].strftime('%d/%m/%Y')})")
else:
    st.subheader("🌡️ Predicción para Mañana")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "🌡️ Temperatura Predicha", 
        f"{temp_predicha:.1f}°C"
    )

with col2:
    st.metric("❄️ Probabilidad Helada", f"{prob_helada:.1f}%")

with col3:
    st.metric("🔎 Nivel de Riesgo", f"{color_riesgo} {riesgo}")

# ============================================================
# ALERTA (SOLO MAÑANA)
# ============================================================
st.markdown("---")
if resultado:
    # Convertir mes a español manualmente
    meses_es = {
        1: 'enero', 2: 'febrero', 3: 'marzo', 4: 'abril',
        5: 'mayo', 6: 'junio', 7: 'julio', 8: 'agosto',
        9: 'septiembre', 10: 'octubre', 11: 'noviembre', 12: 'diciembre'
    }
    
    fecha_pred = resultado['fecha_prediccion']
    dia = fecha_pred.day
    mes = meses_es[fecha_pred.month]
    anio = fecha_pred.year
    fecha_prediccion_str = f"{dia} de {mes} de {anio}"
    
    if temp_predicha <= 0:
        st.error(f"⚠️ **ALERTA DE HELADA**: Se espera temperatura bajo 0°C el **{fecha_prediccion_str}**")
    elif temp_predicha <= 2:
        st.warning(f"⚡ **PRECAUCIÓN**: Temperatura cercana al punto de congelación el **{fecha_prediccion_str}**")
    else:
        st.success(f"✅ No se espera helada para el **{fecha_prediccion_str}**")
else:
    if temp_predicha <= 0:
        st.error(f"⚠️ **ALERTA DE HELADA**: Se espera temperatura bajo 0°C")
    elif temp_predicha <= 2:
        st.warning(f"⚡ **PRECAUCIÓN**: Temperatura cercana al punto de congelación")
    else:
        st.success(f"✅ No se espera helada")

# ============================================================
# MAPA INTERACTIVO CON POLÍGONO DE MADRID
# ============================================================
st.subheader("🗺️ Mapa de Temperatura - Madrid, Cundinamarca")

# Coordenadas de Madrid, Cundinamarca (centro)
madrid_lat = 4.7333
madrid_lon = -74.2667

# Crear mapa
mapa = folium.Map(
    location=[madrid_lat, madrid_lon],
    zoom_start=13,
    tiles='OpenStreetMap'
)

# Solo círculo - sin polígono
folium.Circle(
    location=[madrid_lat, madrid_lon],
    radius=3000,
    color=color_mapa,
    weight=3,
    fill=True,
    fillColor=color_mapa,
    fillOpacity=0.3,
    popup=f"<b>Madrid, Cundinamarca</b><br>Temp. predicha: {temp_predicha:.1f}°C<br>Riesgo: {riesgo}",
    tooltip="Madrid, Cundinamarca"
).add_to(mapa)

# Marcador en el centro con temperatura
folium.Marker(
    location=[madrid_lat, madrid_lon],
    popup=f"<b>Madrid, Cundinamarca</b><br>🌡️ Temperatura predicha: <b>{temp_predicha:.1f}°C</b><br>❄️ Probabilidad helada: <b>{prob_helada:.1f}%</b><br>🔎 Riesgo: <b>{riesgo}</b><br>📅 Fecha: {resultado['fecha_prediccion'] if resultado else 'N/A'}",
    tooltip=f"🌡️ {temp_predicha:.1f}°C - {riesgo}",
    icon=folium.Icon(color='red' if color_mapa == 'red' else 'orange' if color_mapa == 'orange' else 'blue', 
                     icon='thermometer-half', prefix='fa')
).add_to(mapa)

# Mostrar mapa
st_folium(mapa, width=700, height=500)

# ============================================================
# INFORMACIÓN Y FOOTER
# ============================================================
st.markdown("---")

st.info("""
📍 **Sistema de predicción de heladas para Madrid, Cundinamarca**

- 🤖 Modelos: Ridge Regression (temperatura) + Ridge Classifier (heladas)
- 📊 Entrenamiento: 30 años de datos históricos de IDEAM
""")

st.caption(f"🕐 Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
