"""
motores/var/motor.py
=====================
Motor de cálculo de Value at Risk (VaR) para un portafolio de acciones, fondos
mutuos y renta fija (bonos y CD), con tres metodologías sobre la MISMA ventana de escenarios
históricos, para que los resultados sean comparables entre sí:

  1. Histórico    — percentil empírico de la distribución real de P&L.
  2. Monte Carlo   — simulación normal multivariada (conserva correlaciones)
                     a partir de la media y covarianza histórica.
  3. Paramétrico   — delta-normal analítico (z * desviación estándar).

Este módulo NO se conecta a ninguna base de datos: recibe un JSON ya armado
(parámetros + activos + escenarios de precio) y devuelve los resultados.
Quien construye ese JSON con datos reales es el backend Java (Spring), que
ya tiene acceso a los precios y tipos de cambio en PostgreSQL. Así el motor
queda desacoplado y se puede probar con cualquier fuente de datos.

Renta fija (bloque opcional `rentaFija`): cada bono se revalúa en la fecha de valoración con la curva
base más el cambio absoluto de cada plazo entre las MISMAS fechas consecutivas que usan los escenarios de
precio (ver renta_fija.motor.curva_en_fechas). Su variación (VP escenario / VP base - 1, combinada con el
tipo de cambio igual que una acción) entra como una columna más de la matriz de variaciones: así las tres
metodologías y el beneficio de diversificación acciones-bonos salen de un único cálculo.

Origen: adaptado y extendido de "Motor Var Acciones/Motor API/var_api_flask.py"
(motor validado contra la plantilla Excel de VaR de acciones/fondos mutuos).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..renta_fija.motor import ErrorDatosRentaFija, _flujos_df, curva_en_fechas, riesgo_instrumento

METODOLOGIAS_VALIDAS = ("historico", "montecarlo", "parametrico")


class ErrorDatosVaR(Exception):
    """Datos de entrada insuficientes o inconsistentes para calcular VaR."""


@dataclass
class ResultadoMetodo:
    """Resultado de UNA metodología a UN nivel de confianza."""
    metodologia: str
    nivel_confianza: float
    var_diversificado: float
    var_no_diversificado: float
    beneficio_diversificacion: float
    var_individual: Dict[str, float] = field(default_factory=dict)
    var_desagregado: Dict[str, float] = field(default_factory=dict)
    # Expected shortfall: pérdida promedio en la cola que supera el VaR (mismo signo que el VaR)
    cvar_diversificado: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "cvarDiversificado": self.cvar_diversificado,
            "metodologia": self.metodologia,
            "nivelConfianza": self.nivel_confianza,
            "varDiversificado": self.var_diversificado,
            "varNoDiversificado": self.var_no_diversificado,
            "beneficioDiversificacion": self.beneficio_diversificacion,
            "varIndividual": self.var_individual,
            "varDesagregado": self.var_desagregado,
        }


class MotorVaR:
    """
    Calcula VaR para un portafolio, sobre un único conjunto de escenarios
    históricos, con las 3 metodologías y los niveles de confianza pedidos.
    """

    def __init__(self, parametros: dict, activos: List[dict], escenarios: List[dict],
                 renta_fija: Optional[dict] = None):
        renta_fija = renta_fija or {}
        bonos = renta_fija.get("instrumentos") or []
        if len(activos) == 0 and not bonos:
            raise ErrorDatosVaR("El portafolio no tiene activos.")
        if len(escenarios) < 31:
            raise ErrorDatosVaR(
                f"Se requieren al menos 31 escenarios de precio para calcular variaciones "
                f"(se recibieron {len(escenarios)})."
            )

        self.parametros = parametros
        self.activos = activos
        self.horizonte_dias = int(parametros.get("horizonteDias", 1))
        self.moneda_reporte = parametros.get("monedaReporte", "USD")
        self.tc_actual = parametros.get("tcActual", {})

        self.nombres_activos = [a["nombre"] for a in activos]
        self.num_activos = len(activos)

        # ------------------------------------------------------------------
        # 1. Ordenar escenarios del más antiguo al más reciente
        # ------------------------------------------------------------------
        escenarios_ordenados = sorted(escenarios, key=lambda x: x["numero"], reverse=True)
        num_escenarios = len(escenarios_ordenados)

        matriz_precios = np.zeros((num_escenarios, self.num_activos))
        matriz_tc = np.ones((num_escenarios, self.num_activos))

        for i, esc in enumerate(escenarios_ordenados):
            for j, activo in enumerate(activos):
                nombre = activo["nombre"]
                if nombre not in esc.get("precios", {}):
                    raise ErrorDatosVaR(
                        f"El escenario #{esc.get('numero')} no tiene precio para '{nombre}'."
                    )
                matriz_precios[i, j] = esc["precios"][nombre]
                matriz_tc[i, j] = self._tc_escenario(esc, activo["monedaActivo"])

        # ------------------------------------------------------------------
        # 2. Variación combinada (precio + tipo de cambio) por escenario
        # ------------------------------------------------------------------
        variacion_precio = (matriz_precios[1:] / matriz_precios[:-1]) - 1
        variacion_tc = (matriz_tc[1:] / matriz_tc[:-1]) - 1
        self.matriz_variacion = (1 + variacion_precio) * (1 + variacion_tc) - 1
        self.num_variaciones = self.matriz_variacion.shape[0]

        # ------------------------------------------------------------------
        # 3. Exposición (MTM) inicial de cada activo en moneda de reporte
        # ------------------------------------------------------------------
        tc_conversion = np.array([
            self.tc_actual.get(f"{a['monedaActivo']}_{self.moneda_reporte}", 1.0)
            for a in activos
        ])
        # Precio de valorización: el `precioActual` del activo (el backend manda el último disponible, que puede
        # ser posterior a la ventana cuando esta se recorta a las fechas con curva); si no viene, el del último escenario.
        precios_iniciales = np.array([
            float(a["precioActual"]) if a.get("precioActual") is not None else matriz_precios[-1, j]
            for j, a in enumerate(activos)
        ], dtype=float)
        num_acciones = np.array([a["numAcciones"] for a in activos], dtype=float)
        self.mtm_por_activo = precios_iniciales * num_acciones * tc_conversion

        # ------------------------------------------------------------------
        # 3b. Renta fija: una columna de variación por bono, en las mismas fechas
        # ------------------------------------------------------------------
        self.riesgo_renta_fija: List[dict] = []
        self.fecha_valoracion_rf: Optional[str] = None
        if bonos:
            var_bonos, mtm_bonos = self._renta_fija(renta_fija, escenarios_ordenados)
            self.matriz_variacion = np.hstack([self.matriz_variacion, var_bonos])
            self.mtm_por_activo = np.concatenate([self.mtm_por_activo, mtm_bonos])
            self.nombres_activos = self.nombres_activos + [b["isin"] for b in bonos]
            self.num_activos = len(self.nombres_activos)

        self.mtm_total = float(np.sum(self.mtm_por_activo))

        if self.mtm_total <= 0:
            raise ErrorDatosVaR("El valor de mercado (MTM) del portafolio debe ser positivo.")

        # ------------------------------------------------------------------
        # 4. P&L histórico por activo y por escenario (base común a las 3 metodologías)
        # ------------------------------------------------------------------
        self.pl_por_activo_hist = self.mtm_por_activo[np.newaxis, :] * self.matriz_variacion
        self.pl_total_hist = np.sum(self.pl_por_activo_hist, axis=1)

        # Media y covarianza de las variaciones (anualizadas al horizonte pedido)
        self.media_variacion = np.mean(self.matriz_variacion, axis=0)
        # rowvar=False: cada columna es un activo
        self.covarianza_variacion = np.cov(self.matriz_variacion, rowvar=False)
        if self.num_activos == 1:
            # np.cov con 1 sola columna devuelve un escalar; lo normalizamos a matriz 1x1
            self.covarianza_variacion = np.array([[float(self.covarianza_variacion)]])

    def _tc_escenario(self, esc: dict, moneda: str) -> float:
        tc_key = f"{moneda}_{self.moneda_reporte}"
        if esc.get("tcHistorico") and tc_key in esc["tcHistorico"]:
            return esc["tcHistorico"][tc_key]
        return self.tc_actual.get(tc_key, 1.0)

    def _renta_fija(self, bloque: dict, escenarios_ordenados: List[dict]):
        """Variaciones (S-1, B) y MTM en moneda de reporte (B,) de los bonos, alineadas a los escenarios."""
        if any(not e.get("fecha") for e in escenarios_ordenados):
            raise ErrorDatosVaR("Para incluir renta fija cada escenario debe traer su 'fecha'.")
        if not bloque.get("fechaValoracion"):
            raise ErrorDatosVaR("rentaFija.fechaValoracion es obligatoria.")
        try:
            fecha_base = pd.Timestamp(bloque["fechaValoracion"]).normalize()
            fechas = [pd.Timestamp(e["fecha"]).normalize() for e in escenarios_ordenados]
            curvas_raw = bloque.get("curvas") or {}
            curvas = {}
            columnas, mtms = [], []
            for instr in bloque["instrumentos"]:
                for c in ("isin", "moneda", "curva", "nominal"):
                    if c not in instr:
                        raise ErrorDatosVaR(f"Instrumento de renta fija sin '{c}': {instr.get('isin', instr)}")
                codigo = instr["curva"]
                if codigo not in curvas:
                    if codigo not in curvas_raw:
                        raise ErrorDatosVaR(f"Falta la curva '{codigo}' en rentaFija.curvas.")
                    curvas[codigo] = curva_en_fechas(curvas_raw[codigo], fecha_base, fechas, codigo)
                curva = curvas[codigo]

                flujos = _flujos_df({**instr, "tipo": instr.get("tipo", "BONO")}, fecha_base)
                if (flujos["dias"] > 0).sum() == 0:
                    raise ErrorDatosVaR(f"{instr['isin']} está vencido al {fecha_base.date()}.")
                nominal = float(instr["nominal"])
                factores = curva.factor_en(flujos["dias"].to_numpy())                 # (S, F)
                vp = nominal * (factores * (flujos["montoPct"].to_numpy() / 100.0)[None, :]).sum(axis=1)
                vp_base = float(vp[0])
                if vp_base <= 0:
                    raise ErrorDatosVaR(f"{instr['isin']}: valor presente no positivo en la fecha de valoración.")

                # Mismo tratamiento cambiario que una acción: (1 + var. valor) * (1 + var. TC) - 1
                tc = np.array([self._tc_escenario(e, instr["moneda"]) for e in escenarios_ordenados])
                var_tc = tc[1:] / tc[:-1] - 1
                columnas.append((vp[1:] / vp_base) * (1 + var_tc) - 1)
                tc_hoy = self.tc_actual.get(f"{instr['moneda']}_{self.moneda_reporte}", 1.0)
                mtms.append(vp_base * tc_hoy)

                self.riesgo_renta_fija.append({
                    "isin": instr["isin"], "moneda": instr["moneda"], "curva": codigo,
                    "mtmOrigen": vp_base, "mtm": vp_base * tc_hoy,
                    **riesgo_instrumento(flujos, curva, nominal, vp_base),
                })
        except ErrorDatosRentaFija as e:
            raise ErrorDatosVaR(str(e)) from e
        self.fecha_valoracion_rf = fecha_base.strftime("%Y-%m-%d")
        return np.column_stack(columnas), np.array(mtms)

    # ======================================================================
    # Metodología 1: Histórica (percentil empírico)
    # ======================================================================
    def _historico(self, nivel_confianza: float) -> ResultadoMetodo:
        # Cuenta fija de peores escenarios (no escalada por el tamaño de la ventana): así calcula
        # la plantilla Excel validada, que trabaja con una ventana de ~100 escenarios.
        k_index = self._k_index_conteo_fijo(nivel_confianza, self.num_variaciones)

        var_individual: Dict[str, float] = {}
        for i, nombre in enumerate(self.nombres_activos):
            pl_ordenado = np.sort(self.pl_por_activo_hist[:, i])
            var_individual[nombre] = float(pl_ordenado[k_index])

        pl_total_ordenado = np.sort(self.pl_total_hist)
        var_div = float(pl_total_ordenado[k_index])
        cvar_div = float(np.mean(pl_total_ordenado[: k_index + 1]))

        # Escenario donde ocurre el VaR diversificado → VaR desagregado (marginal) por activo
        idx_var = int(np.argmin(np.abs(self.pl_total_hist - var_div)))
        var_desagregado = {
            nombre: float(self.pl_por_activo_hist[idx_var, i])
            for i, nombre in enumerate(self.nombres_activos)
        }

        return self._empaquetar("historico", nivel_confianza, var_div, var_individual, var_desagregado, cvar_div)

    # ======================================================================
    # Metodología 2: Monte Carlo (simulación normal multivariada)
    # ======================================================================
    def _montecarlo(self, nivel_confianza: float, num_simulaciones: int) -> ResultadoMetodo:
        h = self.horizonte_dias
        media_h = self.media_variacion * h
        covarianza_h = self.covarianza_variacion * h

        rng = np.random.default_rng(42)  # semilla fija: resultados reproducibles
        simulaciones = rng.multivariate_normal(media_h, covarianza_h, size=num_simulaciones)

        pl_por_activo_sim = self.mtm_por_activo[np.newaxis, :] * simulaciones
        pl_total_sim = np.sum(pl_por_activo_sim, axis=1)

        # Aquí sí se escala con el tamaño de la muestra (num_simulaciones es grande, p. ej. 10 000):
        # un conteo fijo (como en histórico) daría un percentil equivocado.
        k_index = self._k_index_percentil(nivel_confianza, num_simulaciones)

        var_individual = {
            nombre: float(np.sort(pl_por_activo_sim[:, i])[k_index])
            for i, nombre in enumerate(self.nombres_activos)
        }

        pl_total_ordenado = np.sort(pl_total_sim)
        var_div = float(pl_total_ordenado[k_index])
        cvar_div = float(np.mean(pl_total_ordenado[: k_index + 1]))

        idx_var = int(np.argmin(np.abs(pl_total_sim - var_div)))
        var_desagregado = {
            nombre: float(pl_por_activo_sim[idx_var, i])
            for i, nombre in enumerate(self.nombres_activos)
        }

        return self._empaquetar("montecarlo", nivel_confianza, var_div, var_individual, var_desagregado, cvar_div)

    # ======================================================================
    # Metodología 3: Paramétrico / delta-normal (analítico)
    # ======================================================================
    def _parametrico(self, nivel_confianza: float) -> ResultadoMetodo:
        h = self.horizonte_dias
        z = float(norm.ppf(1 - nivel_confianza))  # negativo: cuantil de pérdida

        w = self.mtm_por_activo
        sigma = self.covarianza_variacion * h
        varianza_portafolio = float(w @ sigma @ w)
        desv_portafolio = float(np.sqrt(max(varianza_portafolio, 0.0)))
        var_div = z * desv_portafolio
        # ES normal: -sigma * phi(z) / (1 - c), con el mismo signo de pérdida que el VaR
        cvar_div = -desv_portafolio * float(norm.pdf(z)) / (1 - nivel_confianza)

        # Asignación de Euler: componente_i = z * w_i * (Σw)_i / desv_portafolio.
        # Por homogeneidad de grado 1 de la desviación estándar, la suma de los componentes
        # da exactamente el VaR diversificado (a diferencia de histórico/Monte Carlo, donde
        # el "desagregado" es el P&L de cada activo en el escenario del peor caso).
        sigma_w = sigma @ w
        if desv_portafolio > 0:
            var_desagregado = {
                nombre: float(z * w[i] * sigma_w[i] / desv_portafolio)
                for i, nombre in enumerate(self.nombres_activos)
            }
        else:
            var_desagregado = {nombre: 0.0 for nombre in self.nombres_activos}

        var_individual = {
            nombre: float(z * w[i] * np.sqrt(max(sigma[i, i], 0.0)))
            for i, nombre in enumerate(self.nombres_activos)
        }

        return self._empaquetar("parametrico", nivel_confianza, var_div, var_individual, var_desagregado, cvar_div)

    # ======================================================================
    # Utilidades comunes
    # ======================================================================
    @staticmethod
    def _k_index_conteo_fijo(nivel_confianza: float, n: int) -> int:
        """
        Índice (orden ascendente, 0-based) del VaR histórico, EXACTAMENTE como la plantilla Excel
        validada: K = 100*(1-confianza) truncado a entero, sin escalar por el tamaño de la muestra.
        Ej.: 97.5% -> 2do peor escenario; 95% -> 5to peor; 99% -> el peor. Se usa `int()` (trunca),
        no `round()`: con floats, 100*(1-0.975) da 2.4999999999999996 o 2.500000000000002 según el
        redondeo binario, y `round()` puede saltar al entero equivocado justo en ese límite.
        """
        k_esimo = 100 * (1 - nivel_confianza)
        return min(max(int(k_esimo) - 1, 0), n - 1)

    @staticmethod
    def _k_index_percentil(nivel_confianza: float, n: int) -> int:
        """Índice del percentil de pérdida escalado al tamaño de la muestra (para Monte Carlo)."""
        k_esimo = 100 * (1 - nivel_confianza)
        return min(max(int(round(n * k_esimo / 100)) - 1, 0), n - 1)

    def _empaquetar(
        self,
        metodologia: str,
        nivel_confianza: float,
        var_div: float,
        var_individual: Dict[str, float],
        var_desagregado: Dict[str, float],
        cvar_div: Optional[float] = None,
    ) -> ResultadoMetodo:
        var_no_div = sum(var_individual.values())
        beneficio = var_no_div - var_div
        return ResultadoMetodo(
            metodologia=metodologia,
            nivel_confianza=nivel_confianza,
            var_diversificado=var_div,
            var_no_diversificado=var_no_div,
            beneficio_diversificacion=beneficio,
            var_individual=var_individual,
            var_desagregado=var_desagregado,
            cvar_diversificado=cvar_div,
        )

    def calcular(
        self,
        metodologias: List[str],
        niveles_confianza: List[float],
        num_simulaciones: int = 10000,
    ) -> dict:
        """Calcula todas las combinaciones (metodología x nivel de confianza) pedidas."""
        inicio = time.time()
        resultados: List[ResultadoMetodo] = []

        for metodologia in metodologias:
            if metodologia not in METODOLOGIAS_VALIDAS:
                raise ErrorDatosVaR(
                    f"Metodología '{metodologia}' no soportada. Use una de: {METODOLOGIAS_VALIDAS}."
                )
            for nivel in niveles_confianza:
                if not 0 < nivel < 1:
                    raise ErrorDatosVaR(f"El nivel de confianza debe estar entre 0 y 1 (recibido {nivel}).")
                if metodologia == "historico":
                    resultados.append(self._historico(nivel))
                elif metodologia == "montecarlo":
                    resultados.append(self._montecarlo(nivel, num_simulaciones))
                else:
                    resultados.append(self._parametrico(nivel))

        tiempo_ms = (time.time() - inicio) * 1000

        salida = {
            "mtmTotal": self.mtm_total,
            "mtmPorActivo": {n: float(v) for n, v in zip(self.nombres_activos, self.mtm_por_activo)},
            "numEscenarios": self.num_variaciones,
            "horizonteDias": self.horizonte_dias,
            "monedaReporte": self.moneda_reporte,
            "tiempoCalculoMs": tiempo_ms,
            "resultados": [r.to_dict() for r in resultados],
        }

        # Serie completa de P&L histórico del portafolio (una por escenario), para graficar el
        # histograma de pérdidas/ganancias. Solo tiene sentido para "histórico" (son escenarios
        # reales con fecha real); Monte Carlo genera miles de simulaciones sintéticas sin fecha.
        if "historico" in metodologias:
            salida["distribucionHistorica"] = self.pl_total_hist.tolist()
        if self.riesgo_renta_fija:
            salida["rentaFija"] = {"fechaValoracion": self.fecha_valoracion_rf, "instrumentos": self.riesgo_renta_fija}

        return salida
