# ============================================================
#  APP DE HELADAS CON PREDICCIÓN REAL
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from datetime import datetime
import joblib
from pathlib import Path

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
# SIDEBAR - CONTROL Y DEBUG
# ============================================================
st.sidebar.header("⚙️ Configuración")

# Botón para actualizar predicción
if st.sidebar.button("🔄 Actualizar Predicción", type="primary"):
    st.cache_resource.clear()
    st.rerun()

st.sidebar.markdown("---")

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
        # Hacer predicción real
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
            # Extraer resultados
            temp_predicha = resultado['temperatura_predicha']
            prob_helada = resultado['probabilidad_helada']
            riesgo = resultado['riesgo']
            color_riesgo = resultado['emoji_riesgo']
            color_mapa = resultado['color_mapa']
            
            # Mostrar en sidebar para debug
            st.sidebar.subheader("🔍 Información de Predicción")
            st.sidebar.write(f"📅 Fecha consulta: {resultado['fecha_consulta']}")
            st.sidebar.write(f"🎯 Predicción para: {resultado['fecha_prediccion']}")
            st.sidebar.write(f"🌡️ Temp. ayer: {resultado['temp_ayer']:.1f}°C")
            st.sidebar.write(f"📊 Cambio esperado: {resultado['cambio_esperado']:.1f}°C")
            st.sidebar.write(f"📈 Promedio 7 días: {resultado['temp_promedio_7d']:.1f}°C")
            st.sidebar.write(f"⬇️ Mínima 7 días: {resultado['temp_minima_7d']:.1f}°C")
            st.sidebar.write(f"⬆️ Máxima 7 días: {resultado['temp_maxima_7d']:.1f}°C")
            
            st.success("✅ Predicción actualizada correctamente")

# ============================================================
# MÉTRICAS PRINCIPALES
# ============================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "🌡️ Temperatura Predicha", 
        f"{temp_predicha:.1f}°C",
        delta=f"{resultado['cambio_esperado']:.1f}°C" if resultado and 'cambio_esperado' in resultado else None
    )

with col2:
    st.metric("❄️ Probabilidad Helada", f"{prob_helada:.1f}%")

with col3:
    st.metric("🔎 Nivel de Riesgo", f"{color_riesgo} {riesgo}")

# ============================================================
# ALERTA
# ============================================================
st.markdown("---")
if temp_predicha <= 0:
    st.error("⚠️ **ALERTA DE HELADA**: Se espera temperatura bajo 0°C mañana")
elif temp_predicha <= 2:
    st.warning("⚡ **PRECAUCIÓN**: Temperatura cercana al punto de congelación")
else:
    st.success("✅ No se espera helada para mañana")

# ============================================================
# MAPA INTERACTIVO
# ============================================================
st.subheader("🗺️ Mapa de Temperatura - Madrid, Cundinamarca")

# Coordenadas de Madrid, Cundinamarca
madrid_lat = 4.7333
madrid_lon = -74.2667

# Crear mapa
mapa = folium.Map(
    location=[madrid_lat, madrid_lon],
    zoom_start=13,
    tiles='OpenStreetMap'
)

# Marcador con temperatura
folium.Marker(
    location=[madrid_lat, madrid_lon],
    popup=f"<b>Madrid, Cundinamarca</b><br>Temperatura predicha: {temp_predicha:.1f}°C<br>Riesgo: {riesgo}",
    tooltip=f"🌡️ {temp_predicha:.1f}°C - {riesgo}",
    icon=folium.Icon(color=color_mapa, icon='thermometer-half', prefix='fa')
).add_to(mapa)

# Círculo de zona afectada
folium.Circle(
    location=[madrid_lat, madrid_lon],
    radius=2000,
    color=color_mapa,
    fill=True,
    fillOpacity=0.2,
    popup="Zona de cobertura"
).add_to(mapa)

# Mostrar mapa
st_folium(mapa, width=700, height=500)

# ============================================================
# HISTORIAL (si hay datos)
# ============================================================
if resultado and PREDICTOR_DISPONIBLE and predictor:
    st.markdown("---")
    st.subheader("📊 Historial de Temperatura (Últimos 30 días)")
    
    historial = resultado['historial_30d']
    
    # Gráfico
    st.line_chart(
        historial.set_index('Fecha')[predictor.target],
        use_container_width=True
    )
    
    # Estadísticas generales
    with st.expander("📈 Ver Estadísticas Generales"):
        stats = predictor.estadisticas_generales()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📅 Registros", stats['total_registros'])
        with col2:
            st.metric("🌡️ Temp. Promedio", f"{stats['temp_promedio']:.1f}°C")
        with col3:
            st.metric("❄️ Heladas Totales", stats['heladas_totales'])
        with col4:
            st.metric("📊 % Heladas", f"{stats['porcentaje_heladas']:.1f}%")

# ============================================================
# INFORMACIÓN Y FOOTER
# ============================================================
st.markdown("---")
st.info("📍 Este sistema utiliza modelos de Machine Learning entrenados con datos históricos de IDEAM para predecir temperaturas y heladas en Madrid, Cundinamarca.")

# Footer
st.caption(f"🕐 Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("💡 Presiona '🔄 Actualizar Predicción' en la barra lateral para recalcular")
