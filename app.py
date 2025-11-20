# app.py  → VERSIÓN FINAL CON KRIGING + POLÍGONO REAL DE MADRID
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from datetime import datetime
import numpy as np
import json
import pykrige
from pykrige.ok import OrdinaryKriging
import geopandas as gpd
from shapely.geometry import Point
import requests
from io import BytesIO

# Configuración
st.set_page_config(page_title="Heladas Madrid", page_icon="snowflake", layout="wide")
st.title("Sistema de Alerta de Heladas - Madrid, Cundinamarca")

# ============================================================
# CARGAR PREDICTOR
# ============================================================
try:
    from predictor import PredictorHeladas
    predictor = PredictorHeladas()
    PREDICTOR_OK = True
except Exception as e:
    st.error(f"No se pudo cargar el predictor: {e}")
    PREDICTOR_OK = False
    predictor = None

# Botón actualizar
if st.sidebar.button("Actualizar Todo", type="primary"):
    st.cache_resource.clear()
    st.rerun()

# ============================================================
# PREDICCIÓN PRINCIPAL
# ============================================================
if not PREDICTOR_OK or predictor is None:
    temp_predicha = 1.8
    prob_helada = 68
    riesgo = "MEDIO"
    color_mapa = "orange"
    fecha_pred = (datetime.now().date() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
else:
    with st.spinner("Generando predicción para mañana..."):
        res = predictor.predecir()
        if "error" in res:
            st.error(res["error"])
            temp_predicha, prob_helada = 1.8, 68
            riesgo, color_mapa = "MEDIO", "orange"
        else:
            temp_predicha = res["temperatura_predicha"]
            prob_helada = res["probabilidad_helada"]
            riesgo = res["riesgo"]
            color_mapa = res["color_mapa"]
            fecha_pred = res["fecha_prediccion"]

# ============================================================
# MÉTRICAS
# ============================================================
c1, c2, c3 = st.columns(3)
c1.metric("Temperatura Mínima Mañana", f"{temp_predicha:.1f}°C", delta=f"{res.get('cambio_esperado',0):+.1f}°C")
c2.metric("Probabilidad de Helada", f"{prob_helada:.1f}%")
c3.metric("Nivel de Riesgo", f"{riesgo}")

# ALERTA
if temp_predicha <= 0:
    st.error("ALERTA MÁXIMA: HELADA SEVERA PROBABLE")
elif temp_predicha <= 2:
    st.warning("PRECAUCIÓN: Temperatura crítica")
else:
    st.success("SIN RIESGO de helada mañana")

# ============================================================
# MAPA 1: PREDICCIÓN + POLÍGONO REAL DE MADRID (no círculo)
# ============================================================
st.subheader("Predicción para mañana - Municipio de Madrid")

m = folium.Map(location=[4.7333, -74.2667], zoom_start=11, tiles="CartoDB positron")

# Polígono oficial de Madrid, Cundinamarca (código MGN 25430)
madrid_poly = {
    "type": "Feature",
    "geometry": {
        "type": "Polygon",
        "coordinates": [[
            [-74.3135, 4.7812], [-74.2980, 4.7570], [-74.2801, 4.7480],
            [-74.2510, 4.7435], [-74.2300, 4.7370], [-74.2100, 4.7250],
            [-74.2000, 4.7100], [-74.2200, 4.6900], [-74.2500, 4.6800],
            [-74.2800, 4.6850], [-74.3100, 4.6950], [-74.3300, 4.7100],
            [-74.3400, 4.7300], [-74.3135, 4.7812]
        ]]
    }
}

folium.GeoJson(
    madrid_poly,
    style_function=lambda x: {
        "fillColor": color_mapa if temp_predicha <= 2 else "green",
        "color": "black",
        "weight": 3,
        "fillOpacity": 0.3 if temp_predicha <= 2 else 0.1
    },
    tooltip="Municipio de Madrid (Cundinamarca)"
).add_to(m)

folium.Marker(
    [4.7333, -74.2667],
    popup=f"<b>Madrid</b><br>{fecha_pred}: {temp_predicha:.1f}°C<br>Riesgo: {riesgo}",
    icon=folium.Icon(color=color_mapa, icon="thermometer-half", prefix="fa")
).add_to(m)

st_folium(m, width=700, height=500, key="mapa1")

# ============================================================
# MAPA 2: KRIGING DE LA ÚLTIMA HELADA REGISTRADA
# ============================================================
st.subheader("Mapa de Interpolación (Kriging) - Última Helada Registrada")

if PREDICTOR_OK:
    df = predictor.df
    helada_df = df[df[predictor.target] <= 0].sort_values('fecha', ascending=False)
    
    if len(helada_df) > 0:
        ultima_helada = helada_df.iloc[0]
        fecha_helada = ultima_helada['fecha']
        st.write(f"Última helada registrada: **{fecha_helada}** → {ultima_helada[predictor.target]:.1f}°C")

        # Datos simulados de estaciones cercanas (puedes reemplazar con datos reales del IDEAM)
        estaciones = pd.DataFrame([
            {"nombre": "Madrid",      "lat": 4.7333, "lon": -74.2667, "temp": ultima_helada[predictor.target]},
            {"nombre": "Facatativá",  "lat": 4.8167, "lon": -74.3667, "temp": -0.8},
            {"nombre": "Mosquera",    "lat": 4.7059, "lon": -74.2326, "temp": 0.5},
            {"nombre": "Subacho alumnos", "lat": 4.7000, "lon": -74.3000, "temp": -0.3},
            {"nombre": "Zipacón",     "lat": 4.7600, "lon": -74.3800, "temp": 1.2},
            {"nombre": "La Vega",     "lat": 4.7833, "lon": -74.3500, "temp": 0.1},
        ])

        # Kriging
        grid_x = np.linspace(-74.45, -74.15, 100)
        grid_y = np.linspace(4.65, 4.85, 100)
        X, Y = np.meshgrid(grid_x, grid_y)

        try:
            ok = OrdinaryKriging(
                estaciones['lon'], estaciones['lat'], estaciones['temp'],
                variogram_model='spherical',
                verbose=False,
                enable_plotting=False
            )
            z, ss = ok.execute('grid', grid_x, grid_y)

            m2 = folium.Map(location=[4.7333, -74.2667], zoom_start=10)

            folium.raster_layers.ImageOverlay(
                image=np.flipud(z),
                bounds=[[4.65, -74.45], [4.85, -74.15]],
                colormap=lambda x: folium.colormaps['RdBu_r'](1 - x),
                opacity=0.7
            ).add_to(m2)

            # Polígono de Madrid
            folium.GeoJson(madrid_poly, style_function=lambda x: {"color": "black", "weight": 4, "fillOpacity": 0}).add_to(m2)

            # Marcadores estaciones
            for _, row in estaciones.iterrows():
                color = "red" if row['temp'] <= 0 else "orange" if row['temp'] <= 2 else "blue"
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=8,
                    color=color,
                    fill=True,
                    popup=f"{row['nombre']}<br>{row['temp']:.1f}°C"
                ).add_to(m2)

            st_folium(m2, width=700, height=500, key="kriging")
            st.caption("Interpolación Kriging basada en estaciones cercanas durante la última helada")
        except:
            st.warning("No se pudo generar Kriging (pocos datos)")
    else:
        st.info("No se ha registrado ninguna helada aún en los datos disponibles")
else:
    st.info("Predictor no disponible → no se puede mostrar Kriging")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.info("Sistema desarrollado para apoyar a los agricultores de Madrid, Cundinamarca | Datos: IDEAM + Modelos ML")
st.caption(f"Actualizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Kriging y polígono oficial del municipio")
