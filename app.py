# ============================================================
#  APP SIMPLE DE HELADAS CON MAPA
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from datetime import datetime

# Configuración
st.set_page_config(
    page_title="Heladas Madrid",
    page_icon="❄️",
    layout="wide"
)

# Título
st.title("❄️ Sistema de Alerta de Heladas - Madrid")

# Simular predicción (reemplaza después con tu modelo)
temp_predicha = 1.5  # Puedes cambiar este número
prob_helada = 65

# Determinar riesgo
if temp_predicha <= 0:
    riesgo = "ALTO"
    color_riesgo = "🔴"
elif temp_predicha <= 2:
    riesgo = "MEDIO"
    color_riesgo = "🟡"
else:
    riesgo = "BAJO"
    color_riesgo = "🟢"

# Mostrar métricas
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🌡️ Temperatura", f"{temp_predicha:.1f}°C")

with col2:
    st.metric("❄️ Probabilidad Helada", f"{prob_helada}%")

with col3:
    st.metric("🔎 Riesgo", f"{color_riesgo} {riesgo}")

# Alerta
if temp_predicha <= 0:
    st.error("⚠️ ALERTA: Se espera helada mañana")
else:
    st.success("✅ No se espera helada")

# MAPA INTERACTIVO
st.subheader("🗺️ Mapa de Temperatura")

# Coordenadas de Madrid, Cundinamarca
madrid_lat = 4.7333
madrid_lon = -74.2667

# Crear mapa
mapa = folium.Map(
    location=[madrid_lat, madrid_lon],
    zoom_start=13
)

# Color según temperatura
if temp_predicha <= 0:
    color = 'red'
elif temp_predicha <= 2:
    color = 'orange'
else:
    color = 'green'

# Marcador
folium.Marker(
    location=[madrid_lat, madrid_lon],
    popup=f"Temperatura: {temp_predicha:.1f}°C",
    tooltip="Madrid",
    icon=folium.Icon(color=color, icon='thermometer', prefix='fa')
).add_to(mapa)

# Círculo de zona
folium.Circle(
    location=[madrid_lat, madrid_lon],
    radius=2000,
    color=color,
    fill=True,
    fillOpacity=0.3
).add_to(mapa)

# Mostrar mapa
st_folium(mapa, width=700, height=500)

# Información
st.info("📍 El mapa muestra la ubicación de Madrid, Cundinamarca con la temperatura predicha")

# Footer
st.markdown("---")
st.caption(f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")