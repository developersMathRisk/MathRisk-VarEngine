"""
motores/backtesting/motor.py
==============================
Motor de Backtesting: prueba retrospectiva (walk-forward, FUERA de muestra) del VaR Histórico,
siguiendo el Art. 27° del Reglamento para la Gestión del Riesgo de Mercado (Resolución SBS N° 4906-2017):
"comparar las pérdidas estimadas por sus modelos de medición de riesgos con los resultados efectivamente
generados, para un período de tiempo determinado".

Para cada día de prueba t, el VaR se estima usando SOLO los `ventana` días anteriores a t (sin ver el
futuro) y se compara contra la pérdida/ganancia REAL ocurrida en t. Es la misma convención de conteo
fijo que usa el motor de VaR para la metodología histórica (ver motores/var/motor.py), aplicada en una
ventana deslizante en vez de una sola vez. Un "exceso" es un día en que la pérdida real superó
(fue peor que) el VaR estimado el día anterior.

La prueba de Kupiec (POF - Proportion of Failures) es el test estadístico estándar para decidir si la
tasa de excesos observada es compatible con la tasa esperada por el nivel de confianza, o si el modelo
está mal calibrado.
"""

from typing import List

import numpy as np


class ErrorDatosBacktesting(Exception):
    """Error de datos de entrada, no un bug del motor."""


CHI2_1_95 = 3.841  # valor crítico de chi-cuadrado (1 grado de libertad, 95% de significancia)


class MotorBacktesting:
    def __init__(self, parametros: dict, activos: List[dict], escenarios: List[dict]):
        if not activos:
            raise ErrorDatosBacktesting("La lista 'activos' está vacía.")

        self.moneda_reporte = parametros["monedaReporte"]
        self.tc_actual = parametros.get("tcActual") or {}

        try:
            self.nivel_confianza = float(parametros["nivelConfianza"])
            self.ventana = int(parametros["ventana"])
        except (KeyError, TypeError, ValueError):
            raise ErrorDatosBacktesting("parametros.nivelConfianza y parametros.ventana son obligatorios y numéricos.")

        if not 0 < self.nivel_confianza < 1:
            raise ErrorDatosBacktesting(f"nivelConfianza debe estar entre 0 y 1 (recibido {self.nivel_confianza}).")
        if self.ventana < 31:
            raise ErrorDatosBacktesting(f"La ventana debe ser de al menos 31 días (recibida {self.ventana}).")

        self.num_activos = len(activos)

        # Más antiguo primero (numero más alto = más antiguo; ver convención del motor de VaR)
        escenarios_ordenados = sorted(escenarios, key=lambda x: x["numero"], reverse=True)
        num_escenarios = len(escenarios_ordenados)
        minimo_requerido = self.ventana + 10
        if num_escenarios < minimo_requerido:
            raise ErrorDatosBacktesting(
                f"Se requieren al menos {minimo_requerido} escenarios para evaluar el backtesting con esta "
                f"ventana de {self.ventana} días (se recibieron {num_escenarios})."
            )

        matriz_precios = np.zeros((num_escenarios, self.num_activos))
        matriz_tc = np.ones((num_escenarios, self.num_activos))
        self.fechas: List[str] = []

        for i, esc in enumerate(escenarios_ordenados):
            self.fechas.append(esc.get("fecha"))
            for j, activo in enumerate(activos):
                nombre = activo["nombre"]
                if nombre not in esc.get("precios", {}):
                    raise ErrorDatosBacktesting(f"El escenario #{esc.get('numero')} no tiene precio para '{nombre}'.")
                matriz_precios[i, j] = esc["precios"][nombre]

                tc_key = f"{activo['monedaActivo']}_{self.moneda_reporte}"
                if esc.get("tcHistorico") and tc_key in esc["tcHistorico"]:
                    matriz_tc[i, j] = esc["tcHistorico"][tc_key]
                else:
                    matriz_tc[i, j] = self.tc_actual.get(tc_key, 1.0)

        variacion_precio = (matriz_precios[1:] / matriz_precios[:-1]) - 1
        variacion_tc = (matriz_tc[1:] / matriz_tc[:-1]) - 1
        # P&L combinado (precio + tipo de cambio) por escenario, misma fórmula que el motor de VaR
        self.matriz_variacion = (1 + variacion_precio) * (1 + variacion_tc) - 1
        self.fechas_variacion = self.fechas[1:]  # self.matriz_variacion[k] corresponde a self.fechas_variacion[k]

        # MTM fijo en el valor de la última fecha disponible: el backtesting evalúa la calidad del
        # modelo de riesgo sobre la cartera ACTUAL, no revalúa posiciones históricas día a día.
        tc_conversion = np.array([
            self.tc_actual.get(f"{a['monedaActivo']}_{self.moneda_reporte}", 1.0) for a in activos
        ])
        precios_iniciales = matriz_precios[-1, :]
        num_acciones = np.array([a["numAcciones"] for a in activos])
        self.mtm_por_activo = precios_iniciales * num_acciones * tc_conversion
        self.mtm_total = float(np.sum(self.mtm_por_activo))
        if self.mtm_total <= 0:
            raise ErrorDatosBacktesting("El valor de mercado (MTM) del portafolio debe ser positivo.")

        self.pl_total_hist = np.sum(self.mtm_por_activo[np.newaxis, :] * self.matriz_variacion, axis=1)

    @staticmethod
    def _k_index_conteo_fijo(nivel_confianza: float, n: int) -> int:
        """Misma convención que el motor de VaR histórico: conteo fijo del k-ésimo peor escenario,
        sin escalar por el tamaño de la ventana (ver motores/var/motor.py)."""
        k_esimo = 100 * (1 - nivel_confianza)
        return min(max(int(k_esimo) - 1, 0), n - 1)

    def calcular(self) -> dict:
        n_total = len(self.pl_total_hist)
        serie = []
        excepciones = 0
        dias_evaluados = 0
        k = self._k_index_conteo_fijo(self.nivel_confianza, self.ventana)

        for i in range(self.ventana, n_total):
            ventana_pnl = self.pl_total_hist[i - self.ventana:i]
            ordenado = np.sort(ventana_pnl)  # ascendente: la pérdida más grande primero
            var_estimado = float(ordenado[k])
            pnl_real = float(self.pl_total_hist[i])
            es_excepcion = bool(pnl_real < var_estimado)

            dias_evaluados += 1
            if es_excepcion:
                excepciones += 1

            serie.append({
                "fecha": self.fechas_variacion[i],
                "varEstimado": var_estimado,
                "pnlReal": pnl_real,
                "esExcepcion": es_excepcion,
            })

        if dias_evaluados == 0:
            raise ErrorDatosBacktesting("No hay suficientes días para evaluar el backtesting con esta ventana.")

        tasa_esperada = 1 - self.nivel_confianza
        tasa_observada = excepciones / dias_evaluados
        estadistico_lr = self._kupiec_lr(dias_evaluados, excepciones, tasa_esperada)

        return {
            "mtmTotal": self.mtm_total,
            "nivelConfianza": self.nivel_confianza,
            "ventana": self.ventana,
            "diasEvaluados": dias_evaluados,
            "excepciones": excepciones,
            "tasaObservada": tasa_observada,
            "tasaEsperada": tasa_esperada,
            "kupiec": {
                "estadisticoLR": estadistico_lr,
                "valorCriticoChi2": CHI2_1_95,
                "modeloAdecuado": estadistico_lr <= CHI2_1_95,
            },
            "serie": serie,
        }

    @staticmethod
    def _kupiec_lr(n: int, x: int, p: float) -> float:
        """Estadístico de razón de verosimilitud de la prueba de Kupiec (POF). H0: la tasa real de
        excesos es `p`. Se rechaza H0 (modelo mal calibrado) si LR > 3.841 (chi-cuadrado, 1 g.l., 95%)."""
        if x == 0:
            log_verosim_nula = n * np.log(1 - p)
            log_verosim_alt = 0.0
        elif x == n:
            log_verosim_nula = n * np.log(p)
            log_verosim_alt = 0.0
        else:
            log_verosim_nula = (n - x) * np.log(1 - p) + x * np.log(p)
            tasa_obs = x / n
            log_verosim_alt = (n - x) * np.log(1 - tasa_obs) + x * np.log(tasa_obs)
        return float(-2 * (log_verosim_nula - log_verosim_alt))
