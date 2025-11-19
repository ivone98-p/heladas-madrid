 ============================================================
#  MÓDULO DE PREDICCIÓN PARA STREAMLIT
# ============================================================

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta

class PredictorHeladas:
    """
    Clase que maneja las predicciones de temperatura y heladas
    """
    
    def __init__(self):
        """Inicializa el predictor cargando modelos y datos"""
        self.base_dir = Path('Datos')
        self.modelos_dir = self.base_dir / 'modelos_entrenados'
        self.datos_dir = self.base_dir / 'datos_imputados'
        
        # Cargar modelos
        self._cargar_modelos()
        
        # Cargar datos
        self._cargar_datos()
        
        # Variable para cachear predicción
        self._ultima_prediccion = None
        self._fecha_ultima_prediccion = None
    
    def _cargar_modelos(self):
        """Carga todos los modelos entrenados"""
        try:
            self.modelo_temp = joblib.load(self.modelos_dir / 'modelo_temperatura_ridge.pkl')
            self.scaler_temp = joblib.load(self.modelos_dir / 'scaler_temperatura.pkl')
            self.features_temp = joblib.load(self.modelos_dir / 'features_temperatura.pkl')
            
            self.modelo_helada = joblib.load(self.modelos_dir / 'modelo_helada_ridge.pkl')
            self.scaler_helada = joblib.load(self.modelos_dir / 'scaler_helada.pkl')
            self.features_helada = joblib.load(self.modelos_dir / 'features_helada.pkl')
            
            print("✅ Modelos cargados correctamente")
        except Exception as e:
            print(f"❌ Error cargando modelos: {e}")
            raise
    
    def _cargar_datos(self):
        """Carga datos históricos"""
        try:
            self.df = pd.read_csv(self.datos_dir / 'cundinamarca_imputado_v1.csv')
            self.df['Fecha'] = pd.to_datetime(self.df['Fecha'])
            self.target = 'TMin_21205880_FLORES_CHIBCHA_MADRID'
            
            # Identificar columnas
            self.columnas_prec = [col for col in self.df.columns if 'PREC' in col]
            self.columnas_tmax = [col for col in self.df.columns if 'TMax' in col]
            
            print(f"✅ Datos cargados: {len(self.df)} registros")
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            raise
    
    def _crear_features_temperatura(self, df_input):
        """Crea features para predicción de temperatura (Solo Madrid)"""
        df_out = df_input.copy()
        
        # Temporales
        df_out['Mes'] = df_out['Fecha'].dt.month
        df_out['DíaAño'] = df_out['Fecha'].dt.dayofyear
        df_out['Trimestre'] = df_out['Fecha'].dt.quarter
        df_out['DiaSemana'] = df_out['Fecha'].dt.dayofweek
        df_out['Semana'] = df_out['Fecha'].dt.isocalendar().week
        
        # Cíclicos
        df_out['Mes_sin'] = np.sin(2 * np.pi * df_out['Mes'] / 12)
        df_out['Mes_cos'] = np.cos(2 * np.pi * df_out['Mes'] / 12)
        df_out['DíaAño_sin'] = np.sin(2 * np.pi * df_out['DíaAño'] / 365)
        df_out['DíaAño_cos'] = np.cos(2 * np.pi * df_out['DíaAño'] / 365)
        df_out['Semana_sin'] = np.sin(2 * np.pi * df_out['Semana'] / 52)
        df_out['Semana_cos'] = np.cos(2 * np.pi * df_out['Semana'] / 52)
        df_out['DiaSemana_sin'] = np.sin(2 * np.pi * df_out['DiaSemana'] / 7)
        df_out['DiaSemana_cos'] = np.cos(2 * np.pi * df_out['DiaSemana'] / 7)
        
        # Rezagos
        for lag in [1, 2, 3, 7, 14, 21, 30]:
            df_out[f'TMIN_lag_{lag}'] = df_out[self.target].shift(lag)
        
        # Promedios móviles
        for window in [3, 7, 14, 30]:
            df_out[f'TMIN_ma_{window}'] = df_out[self.target].shift(1).rolling(window=window).mean()
            df_out[f'TMIN_std_{window}'] = df_out[self.target].shift(1).rolling(window=window).std()
            df_out[f'TMIN_min_{window}'] = df_out[self.target].shift(1).rolling(window=window).min()
            df_out[f'TMIN_max_{window}'] = df_out[self.target].shift(1).rolling(window=window).max()
        
        # Diferencias
        df_out['TMIN_diff_1'] = df_out[self.target].diff(1)
        df_out['TMIN_diff_7'] = df_out[self.target].diff(7)
        df_out['TMIN_diff_30'] = df_out[self.target].diff(30)
        
        # Tendencias
        def calcular_tendencia(serie):
            if len(serie) < 5 or serie.isna().all():
                return 0
            try:
                x = np.arange(len(serie))
                coef = np.polyfit(x, serie.values, 1)[0]
                return coef
            except:
                return 0
        
        for window in [7, 14, 30]:
            df_out[f'TMIN_tendencia_{window}'] = df_out[self.target].shift(1).rolling(window=window).apply(
                calcular_tendencia, raw=False
            )
        
        # Rangos
        for window in [7, 14, 30]:
            df_out[f'TMIN_rango_{window}'] = df_out[f'TMIN_max_{window}'] - df_out[f'TMIN_min_{window}']
        
        # Percentiles
        for window in [7, 14, 30]:
            df_out[f'TMIN_q25_{window}'] = df_out[self.target].shift(1).rolling(window=window).quantile(0.25)
            df_out[f'TMIN_q75_{window}'] = df_out[self.target].shift(1).rolling(window=window).quantile(0.75)
        
        # Aceleración
        df_out['TMIN_aceleracion'] = df_out['TMIN_diff_1'].diff(1)
        
        return df_out
    
    def _crear_features_helada(self, df_input):
        """Crea features para predicción de helada (Madrid + PREC + TMax)"""
        df_out = self._crear_features_temperatura(df_input)
        
        # Precipitación
        if len(self.columnas_prec) > 0:
            for col in self.columnas_prec:
                df_out[f'{col}_lag1'] = df_out[col].shift(1)
            
            cols_prec_lag = [f'{col}_lag1' for col in self.columnas_prec]
            df_out['PREC_promedio'] = df_out[cols_prec_lag].mean(axis=1)
            df_out['PREC_max'] = df_out[cols_prec_lag].max(axis=1)
            df_out['PREC_std'] = df_out[cols_prec_lag].std(axis=1)
            
            for lag in [2, 3, 7]:
                df_out[f'PREC_promedio_lag{lag}'] = df_out['PREC_promedio'].shift(lag)
            
            for window in [3, 7, 14]:
                df_out[f'PREC_suma_{window}'] = df_out['PREC_promedio'].shift(1).rolling(window=window).sum()
        
        # Temperatura máxima
        if len(self.columnas_tmax) > 0:
            for col in self.columnas_tmax:
                df_out[f'{col}_lag1'] = df_out[col].shift(1)
            
            cols_tmax_lag = [f'{col}_lag1' for col in self.columnas_tmax]
            df_out['TMAX_promedio'] = df_out[cols_tmax_lag].mean(axis=1)
            df_out['TMAX_std'] = df_out[cols_tmax_lag].std(axis=1)
            df_out['Rango_termico_lag1'] = df_out['TMAX_promedio'] - df_out['TMIN_lag_1']
            
            for window in [3, 7, 14]:
                df_out[f'TMAX_ma_{window}'] = df_out['TMAX_promedio'].shift(1).rolling(window=window).mean()
            
            df_out['TMAX_diff_1'] = df_out['TMAX_promedio'].diff(1)
        
        # Interacciones
        if 'TMAX_promedio' in df_out.columns and 'TMIN_lag_1' in df_out.columns:
            df_out['TMax_TMin_ratio'] = df_out['TMAX_promedio'] / (df_out['TMIN_lag_1'].abs() + 1)
        
        if 'PREC_promedio' in df_out.columns:
            df_out['PREC_binaria'] = (df_out['PREC_promedio'] > 0).astype(int)
        
        # Eliminar columnas originales
        for col in self.columnas_prec + self.columnas_tmax:
            if col in df_out.columns:
                df_out.drop(col, axis=1, inplace=True)
        
        return df_out
    
    def predecir(self, fecha_consulta=None):
        """
        Realiza predicción para el día siguiente
        
        Args:
            fecha_consulta: datetime o None (usa última fecha disponible)
            
        Returns:
            dict con temperatura, probabilidad, riesgo, etc.
        """
        # Si ya predijimos hoy, devolver resultado cacheado
        hoy = datetime.now().date()
        if self._fecha_ultima_prediccion == hoy and self._ultima_prediccion:
            print("📌 Usando predicción cacheada de hoy")
            return self._ultima_prediccion
        
        try:
            # Fecha de consulta
            if fecha_consulta is None:
                fecha_consulta = self.df['Fecha'].max()
            else:
                fecha_consulta = pd.to_datetime(fecha_consulta)
            
            # Datos hasta la fecha de consulta
            df_hasta_hoy = self.df[self.df['Fecha'] <= fecha_consulta].copy()
            
            if len(df_hasta_hoy) < 50:
                return {"error": "Datos insuficientes"}
            
            # ============================================
            # PREDICCIÓN DE TEMPERATURA
            # ============================================
            df_temp = df_hasta_hoy[['Fecha', self.target]].copy()
            df_temp = df_temp.dropna(subset=[self.target])
            df_temp = self._crear_features_temperatura(df_temp)
            df_temp = df_temp.dropna().reset_index(drop=True)
            
            if len(df_temp) == 0:
                return {"error": "No se pudieron crear features de temperatura"}
            
            # Obtener última fila
            ultima_fila = df_temp.iloc[[-1]]
            X_temp = ultima_fila[self.features_temp]
            X_temp_scaled = self.scaler_temp.transform(X_temp)
            
            # Predecir
            temp_predicha = self.modelo_temp.predict(X_temp_scaled)[0]
            
            # ============================================
            # PREDICCIÓN DE HELADA
            # ============================================
            df_helada = df_hasta_hoy[['Fecha', self.target] + self.columnas_prec + self.columnas_tmax].copy()
            df_helada = df_helada.dropna(subset=[self.target])
            
            # Rellenar NaN en PREC y TMax
            for col in self.columnas_prec + self.columnas_tmax:
                if df_helada[col].isna().any():
                    df_helada[col].fillna(df_helada[col].mean(), inplace=True)
            
            df_helada = self._crear_features_helada(df_helada)
            df_helada = df_helada.dropna().reset_index(drop=True)
            
            if len(df_helada) == 0:
                return {"error": "No se pudieron crear features de helada"}
            
            # Obtener última fila
            ultima_fila_helada = df_helada.iloc[[-1]]
            X_helada = ultima_fila_helada[self.features_helada]
            X_helada_scaled = self.scaler_helada.transform(X_helada)
            
            # Predecir
            score_helada = self.modelo_helada.decision_function(X_helada_scaled)[0]
            
            # Convertir score a probabilidad (0-100%)
            probabilidad_helada = 1 / (1 + np.exp(-score_helada))
            probabilidad_helada = probabilidad_helada * 100
            
            helada_predicha = 1 if score_helada > 0 else 0
            
            # ============================================
            # DETERMINAR RIESGO
            # ============================================
            if temp_predicha <= -2:
                riesgo = "MUY ALTO"
                emoji_riesgo = "🔴"
                color_mapa = "red"
            elif temp_predicha <= 0:
                riesgo = "ALTO"
                emoji_riesgo = "🟠"
                color_mapa = "orange"
            elif temp_predicha <= 2:
                riesgo = "MEDIO"
                emoji_riesgo = "🟡"
                color_mapa = "yellow"
            elif temp_predicha <= 4:
                riesgo = "BAJO"
                emoji_riesgo = "🟢"
                color_mapa = "green"
            else:
                riesgo = "MUY BAJO"
                emoji_riesgo = "🟢"
                color_mapa = "green"
            
            # ============================================
            # DATOS ADICIONALES
            # ============================================
            temp_ayer = df_hasta_hoy[self.target].iloc[-1]
            temp_promedio_7d = df_hasta_hoy[self.target].iloc[-7:].mean()
            temp_minima_7d = df_hasta_hoy[self.target].iloc[-7:].min()
            temp_maxima_7d = df_hasta_hoy[self.target].iloc[-7:].max()
            
            # Historial últimos 30 días
            historial_30d = df_hasta_hoy[['Fecha', self.target]].tail(30)
            
            # Resultado
            resultado = {
                "fecha_consulta": fecha_consulta.date(),
                "fecha_prediccion": (fecha_consulta + pd.Timedelta(days=1)).date(),
                "temperatura_predicha": float(temp_predicha),
                "probabilidad_helada": float(probabilidad_helada),
                "helada_predicha": int(helada_predicha),
                "riesgo": riesgo,
                "emoji_riesgo": emoji_riesgo,
                "color_mapa": color_mapa,
                "temp_ayer": float(temp_ayer),
                "temp_promedio_7d": float(temp_promedio_7d),
                "temp_minima_7d": float(temp_minima_7d),
                "temp_maxima_7d": float(temp_maxima_7d),
                "cambio_esperado": float(temp_predicha - temp_ayer),
                "historial_30d": historial_30d,
                "timestamp": datetime.now()
            }
            
            # Cachear predicción
            self._ultima_prediccion = resultado
            self._fecha_ultima_prediccion = hoy
            
            return resultado
            
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def obtener_historial(self, dias=30):
        """Obtiene historial de temperatura"""
        return self.df[['Fecha', self.target]].tail(dias)
    
    def estadisticas_generales(self):
        """Retorna estadísticas del dataset"""
        heladas_totales = (self.df[self.target] <= 0).sum()
        
        return {
            "total_registros": len(self.df),
            "fecha_inicio": self.df['Fecha'].min().date(),
            "fecha_fin": self.df['Fecha'].max().date(),
            "temp_promedio": float(self.df[self.target].mean()),
            "temp_minima": float(self.df[self.target].min()),
            "temp_maxima": float(self.df[self.target].max()),
            "heladas_totales": int(heladas_totales),
            "porcentaje_heladas": float(heladas_totales / len(self.df) * 100)
        }