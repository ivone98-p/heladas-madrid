# ============================================================
#  APP DE HELADAS CON PREDICCIÓN DE 1 DÍA + SECCIÓN 7 DÍAS
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta

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
        predicciones_7dias = []
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
            predicciones_7dias = []
        else:
            # Extraer resultados del PRIMER DÍA (mañana)
            temp_predicha = resultado['temperatura_predicha']
            prob_helada = resultado['probabilidad_helada']
            riesgo = resultado['riesgo']
            color_riesgo = resultado['emoji_riesgo']
            color_mapa = resultado['color_mapa']
            
            # Extraer predicciones de 7 días
            predicciones_7dias = resultado.get('predicciones_7dias', [])
            
            # Mostrar en sidebar para debug (SOLO MAÑANA)
            st.sidebar.subheader("🔍 Información de Predicción")
            st.sidebar.write(f"📅 Fecha de consulta: **{resultado['fecha_consulta']}**")
            st.sidebar.write(f"🎯 Predicción para: **{resultado['fecha_prediccion']}**")
            st.sidebar.write(f"🌡️ Temp. registrada el {resultado['fecha_consulta']}: {resultado['temp_ayer']:.1f}°C")
            st.sidebar.write(f"📊 Cambio esperado: {resultado['cambio_esperado']:.1f}°C")
            st.sidebar.write(f"📈 Promedio 7 días: {resultado['temp_promedio_7d']:.1f}°C")
            st.sidebar.write(f"⬇️ Mínima 7 días: {resultado['temp_minima_7d']:.1f}°C")
            st.sidebar.write(f"⬆️ Máxima 7 días: {resultado['temp_maxima_7d']:.1f}°C")
            
            # Mostrar si se usaron datos simulados
            if resultado.get('datos_simulados', False):
                st.sidebar.warning(f"⚠️ Datos simulados desde {resultado['ultima_fecha_real']}")
            else:
                st.sidebar.success("✅ Usando datos reales completos")
            
            st.success(f"✅ Predicción generada para **{resultado['fecha_prediccion']}**")

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
        f"{temp_predicha:.1f}°C",
        delta=f"{resultado['cambio_esperado']:.1f}°C" if resultado and 'cambio_esperado' in resultado else None
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
    fecha_prediccion_str = resultado['fecha_prediccion'].strftime('%d de %B de %Y')
    
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

# Polígono de Madrid
madrid_polygon_coords = [
    [4.803356, -74.269926],
    [4.803358, -74.265904],
    [4.806183, -74.258581],
    [4.809309, -74.249509],
    [4.812917, -74.238273],
    [4.815928, -74.228919],
    [4.818368, -74.213654],
    [4.816725, -74.198732],
    [4.812413, -74.186883],
    [4.805634, -74.176499],
    [4.797616, -74.169378],
    [4.788518, -74.164436],
    [4.779159, -74.162547],
    [4.769147, -74.163698],
    [4.758664, -74.167542],
    [4.748629, -74.173821],
    [4.739428, -74.181936],
    [4.731416, -74.191421],
    [4.724889, -74.201782],
    [4.720095, -74.212575],
    [4.717198, -74.223392],
    [4.716279, -74.234866],
    [4.717350, -74.246651],
    [4.720364, -74.257439],
    [4.725222, -74.267042],
    [4.731770, -74.275364],
    [4.739798, -74.282358],
    [4.749050, -74.288030],
    [4.759236, -74.292442],
    [4.770038, -74.295711],
    [4.781116, -74.298001],
    [4.792119, -74.299526],
    [4.803356, -74.269926]
]

# Agregar polígono de Madrid
folium.Polygon(
    locations=madrid_polygon_coords,
    color=color_mapa,
    weight=3,
    fill=True,
    fillColor=color_mapa,
    fillOpacity=0.2,
    popup=f"<b>Madrid, Cundinamarca</b><br>Área municipal<br>Temp. predicha: {temp_predicha:.1f}°C<br>Riesgo: {riesgo}",
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

# Círculo de zona urbana central
folium.Circle(
    location=[madrid_lat, madrid_lon],
    radius=1500,
    color=color_mapa,
    weight=2,
    fill=True,
    fillOpacity=0.15,
    popup="Zona urbana central de Madrid",
    tooltip="Centro urbano"
).add_to(mapa)

# Mostrar mapa
st_folium(mapa, width=700, height=500)

# ============================================================
# NUEVA SECCIÓN: PRONÓSTICO EXTENDIDO 7 DÍAS
# ============================================================
if predicciones_7dias and len(predicciones_7dias) > 0:
    st.markdown("---")
    
    # Cards individuales
    with st.expander("🗓️ Ver detalles día por día", expanded=True):
        cols = st.columns(4)
        
        for i, pred in enumerate(predicciones_7dias[:4]):
            with cols[i]:
                st.markdown(f"**{pred['fecha'].strftime('%a %d/%m')}**")
                st.metric("Temp", f"{pred['temperatura']:.1f}°C")
                st.write(f"{pred['emoji']} {pred['riesgo']}")
                st.write(f"Helada: {pred['probabilidad_helada']:.0f}%")
        
        if len(predicciones_7dias) > 4:
            cols2 = st.columns(3)
            for i, pred in enumerate(predicciones_7dias[4:]):
                with cols2[i]:
                    st.markdown(f"**{pred['fecha'].strftime('%a %d/%m')}**")
                    st.metric("Temp", f"{pred['temperatura']:.1f}°C")
                    st.write(f"{pred['emoji']} {pred['riesgo']}")
                    st.write(f"Helada: {pred['probabilidad_helada']:.0f}%")

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

# Info box con fechas
if resultado:
    stats = predictor.estadisticas_generales()
    st.info(f"""
    📍 **Sistema de predicción de heladas para Madrid, Cundinamarca**
    
    - 📅 Datos históricos reales: **{stats['fecha_inicio']} a {resultado['ultima_fecha_real']}**
    - 🎯 Predicción principal: **{resultado['fecha_prediccion'].strftime('%d de %B de %Y')}**
    - 📊 Pronóstico extendido: **7 días** (predicción recursiva)
    - 🤖 Modelos: Ridge Regression (temperatura) + Ridge Classifier (heladas)
    - 📊 Dataset: 30 años de datos históricos de IDEAM
    - 🧠 Entrenamiento: {stats['total_registros']} días con {stats['heladas_totales']} heladas registradas
    {f"- ⚠️ **Datos simulados** desde {resultado['ultima_fecha_real']} hasta {resultado['fecha_consulta']}" if resultado.get('datos_simulados') else ""}
    
    💡 **Nota**: La predicción recursiva de 7 días usa cada día predicho como base para el siguiente. La precisión disminuye con días más lejanos.
    """)
else:
    st.info("📍 Este sistema utiliza modelos de Machine Learning entrenados con 30 años de datos históricos de IDEAM para predecir temperaturas y heladas en Madrid, Cundinamarca.")

# Footer
st.caption(f"🕐 Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("💡 Presiona '🔄 Actualizar Predicción' en la barra lateral para recalcular")
st.caption(f"🤖 Predicción basada en modelos ML" + (f" (datos simulados desde {resultado['ultima_fecha_real']})" if resultado and resultado.get('datos_simulados') else ""))
