#============================================================
#  MÓDULO DE PREDICCIÓN SIMPLIFICADO
# ============================================================

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta

class PredictorHeladas:
    """
    Clase que maneja las predicciones de temperatura y heladas
    Con capacidad de simular datos faltantes hasta hoy
    """
    
    def __init__(self, data_path='Datos'):
        """
        Inicializa el predictor cargando modelos y datos
        
        Args:
            data_path: Ruta base de datos (por defecto 'Datos')
        """
        # Configurar rutas base
        self.base_dir = Path(data_path)
        self.modelos_dir = self.base_dir / 'modelos_entrenados'
        self.datos_dir = self.base_dir / 'datos_imputados'
        
        print(f"📂 Directorio base: {self.base_dir.absolute()}")
        print(f"📂 Directorio modelos: {self.modelos_dir.absolute()}")
        print(f"📂 Directorio datos: {self.datos_dir.absolute()}")
        
        # Verificar que existen los directorios
        if not self.modelos_dir.exists():
            raise FileNotFoundError(f"❌ No se encuentra el directorio de modelos: {self.modelos_dir}")
        if not self.datos_dir.exists():
            raise FileNotFoundError(f"❌ No se encuentra el directorio de datos: {self.datos_dir}")
        
        # Cargar modelos
        self._cargar_modelos()
        
        # Cargar datos
        self._cargar_datos()
        
        # Variable para cachear predicción
        self._ultima_prediccion = None
    
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
            raise Exception(f"❌ Error cargando modelos: {e}")
    
    def _cargar_datos(self):
        """Carga datos históricos"""
        try:
            csv_path = self.datos_dir / 'cundinamarca_imputado_v1.csv'
            
            if not csv_path.exists():
                raise FileNotFoundError(f"❌ No se encuentra el archivo: {csv_path}")
            
            self.df = pd.read_csv(csv_path)
            self.df['Fecha'] = pd.to_datetime(self.df['Fecha'])
            self.target = 'TMin_21205880_FLORES_CHIBCHA_MADRID'
            
            # Identificar columnas
            self.columnas_prec = [col for col in self.df.columns if 'PREC' in col]
            self.columnas_tmax = [col for col in self.df.columns if 'TMax' in col]
            
            print(f"✅ Datos cargados: {len(self.df)} registros")
            print(f"📅 Rango de fechas: {self.df['Fecha'].min().date()} hasta {self.df['Fecha'].max().date()}")
        except Exception as e:
            raise Exception(f"❌ Error cargando datos: {e}")
    
    def _simular_datos_faltantes(self, fecha_hasta):
        """
        Simula datos faltantes desde la última fecha del CSV hasta fecha_hasta
        usando promedios históricos del mismo mes/día
        
        Args:
            fecha_hasta: datetime hasta donde extender
            
        Returns:
            DataFrame extendido
        """
        df_extendido = self.df.copy()
        ultima_fecha = df_extendido['Fecha'].max()
        
        if fecha_hasta <= ultima_fecha:
            return df_extendido
        
        print(f"🔧 Simulando datos desde {ultima_fecha.date()} hasta {fecha_hasta.date()}...")
        
        # Calcular promedios históricos por mes y día
        df_extendido['Mes'] = df_extendido['Fecha'].dt.month
        df_extendido['Dia'] = df_extendido['Fecha'].dt.day
        
        promedios_temp = df_extendido.groupby(['Mes', 'Dia'])[self.target].mean()
        
        promedios_prec = {}
        for col in self.columnas_prec:
            promedios_prec[col] = df_extendido.groupby(['Mes', 'Dia'])[col].mean()
        
        promedios_tmax = {}
        for col in self.columnas_tmax:
            promedios_tmax[col] = df_extendido.groupby(['Mes', 'Dia'])[col].mean()
        
        # Generar fechas faltantes
        fechas_faltantes = pd.date_range(
            start=ultima_fecha + pd.Timedelta(days=1),
            end=fecha_hasta,
            freq='D'
        )
        
        nuevas_filas = []
        for fecha in fechas_faltantes:
            mes = fecha.month
            dia = fecha.day
            
            nueva_fila = {'Fecha': fecha}
            
            # Temperatura mínima (con variación aleatoria pequeña)
            if (mes, dia) in promedios_temp.index:
                temp_base = promedios_temp.loc[(mes, dia)]
                nueva_fila[self.target] = temp_base + np.random.normal(0, 0.5)
            else:
                # Si no hay datos exactos, usar promedio del mes
                temp_mes = df_extendido[df_extendido['Mes'] == mes][self.target].mean()
                nueva_fila[self.target] = temp_mes + np.random.normal(0, 0.5)
            
            # Precipitación
            for col in self.columnas_prec:
                if (mes, dia) in promedios_prec[col].index:
                    nueva_fila[col] = promedios_prec[col].loc[(mes, dia)]
                else:
                    nueva_fila[col] = df_extendido[df_extendido['Mes'] == mes][col].mean()
            
            # Temperatura máxima
            for col in self.columnas_tmax:
                if (mes, dia) in promedios_tmax[col].index:
                    nueva_fila[col] = promedios_tmax[col].loc[(mes, dia)]
                else:
                    nueva_fila[col] = df_extendido[df_extendido['Mes'] == mes][col].mean()
            
            nuevas_filas.append(nueva_fila)
        
        # Agregar filas simuladas
        df_nuevas_filas = pd.DataFrame(nuevas_filas)
        df_extendido = pd.concat([df_extendido, df_nuevas_filas], ignore_index=True)
        df_extendido = df_extendido.sort_values('Fecha').reset_index(drop=True)
        
        # Eliminar columnas auxiliares
        df_extendido = df_extendido.drop(columns=['Mes', 'Dia'], errors='ignore')
        
        print(f"✅ Datos simulados: {len(nuevas_filas)} días agregados")
        
        return df_extendido
    
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
    
    def predecir(self, fecha_consulta=None, forzar_recalculo=False):
        """
        Realiza predicción para el día siguiente usando Machine Learning
        Simula automáticamente datos faltantes hasta hoy
        
        Args:
            fecha_consulta: datetime o None (usa fecha actual)
            forzar_recalculo: bool, si True recalcula aunque haya caché
            
        Returns:
            dict con temperatura, probabilidad, riesgo, etc.
        """
        # Si ya hay predicción cacheada y no se fuerza recálculo
        if self._ultima_prediccion and not forzar_recalculo:
            print("📌 Usando predicción cacheada")
            return self._ultima_prediccion
        
        try:
            # Fecha de consulta: HOY (fecha actual del sistema)
            if fecha_consulta is None:
                fecha_consulta = pd.Timestamp.now().normalize()
            else:
                fecha_consulta = pd.to_datetime(fecha_consulta).normalize()
            
            # SIMULAR datos faltantes hasta HOY
            df_completo = self._simular_datos_faltantes(fecha_consulta)
            
            # Datos hasta la fecha de consulta
            df_hasta_hoy = df_completo[df_completo['Fecha'] <= fecha_consulta].copy()
            
            if len(df_hasta_hoy) < 50:
                return {"error": "Datos insuficientes para predicción"}
            
            print(f"🔮 Generando predicción ML con {len(df_hasta_hoy)} registros históricos...")
            print(f"📅 Fecha de consulta: {fecha_consulta.date()}")
            print(f"🎯 Predicción para: {(fecha_consulta + pd.Timedelta(days=1)).date()}")
            
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
            
            # Predecir con modelo ML
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
            
            # Predecir con modelo ML
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
            
            # Detectar si se usaron datos simulados
            ultima_fecha_real = self.df['Fecha'].max()
            datos_simulados = fecha_consulta > ultima_fecha_real
            
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
                "timestamp": datetime.now(),
                "registros_usados": len(df_hasta_hoy),
                "datos_simulados": datos_simulados,
                "ultima_fecha_real": ultima_fecha_real.date()
            }
            
            # Cachear predicción
            self._ultima_prediccion = resultado
            
            print(f"✅ Predicción ML completada: {temp_predicha:.1f}°C para {resultado['fecha_prediccion']}")
            if datos_simulados:
                print(f"⚠️ Se usaron datos simulados desde {ultima_fecha_real.date()}")
            
            return resultado
            
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def estadisticas_generales(self):
        """Retorna estadísticas del dataset completo"""
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
