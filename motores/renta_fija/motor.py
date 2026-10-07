"""
motores/renta_fija/motor.py
=============================
Motor de valorización y VaR de renta fija (certificados de depósito y bonos) por simulación
histórica de curvas y tipo de cambio. Portado del motor "PrometheusModelos / VaR - 190926" y
reescrito sin estado ni acceso a archivos/BD: todo llega por JSON y se devuelve por JSON.

Método (igual al original):
  * Escenario de curva en la fecha t, para cada plazo p:
        tasa_esc(t, p) = max(0, tasa_base(p) + [tasa(t, p) - tasa(t-1, p)])      (cambio absoluto, en %)
        factor(t, p)   = 1 / (1 + tasa_esc / 100) ** (p / 360)
    La fecha del portafolio no se perturba (variación 0).
  * Cada flujo se descuenta con el factor interpolado linealmente entre los dos plazos de la curva
    que lo encierran. CD: un flujo (nominal) al vencimiento. Bono: flujos = % del nominal.
  * Efecto cambiario: retorno logarítmico diario del TC; TC_esc(t) = TC_base * exp(retorno).
    P&L en moneda funcional = VP_origen(t) * TC_esc(t) - VP_origen(base) * TC_base.
  * VaR / CVaR: percentil (1 - confianza) del P&L simulado del portafolio; se reportan en positivo.

Diferencias deliberadas respecto al original (documentadas en tests/):
  * El TC se une a la curva por FECHA (el original lo hacía por posición de fila).
  * Un flujo fuera del rango de plazos de la curva usa la tasa del extremo con su plazo real
    (el original le asignaba factor 0 en la valorización por escenario y el del extremo en el riesgo).
  * La TIR se resuelve con los plazos reales (act/360) en vez de `numpy_financial.irr` sobre períodos
    equiespaciados; la duración se reporta como Macaulay y como modificada (el original llamaba
    "modificada" a la de Macaulay).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

BASE_DIAS = 360.0
TIPOS_VALIDOS = ("CD", "BONO")


class ErrorDatosRentaFija(Exception):
    """Datos de entrada inconsistentes o insuficientes (se traduce a HTTP 422)."""


def _fecha(valor, campo: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(valor).normalize()
    except Exception as e:  # noqa: BLE001
        raise ErrorDatosRentaFija(f"Fecha inválida en '{campo}': {valor!r}") from e


# ======================================================================
# Factores (escenarios) de curva y tipo de cambio
# ======================================================================
@dataclass
class EscenariosCurva:
    """Matriz de factores de descuento por escenario para una curva."""

    plazos: np.ndarray          # (P,) plazos en días, ascendentes
    fechas: pd.DatetimeIndex    # (S,) fecha de cada escenario
    tasas: np.ndarray           # (S, P) tasa escenario en %
    factores: np.ndarray        # (S, P) factor de descuento
    idx_base: int               # fila del escenario = fecha del portafolio

    def factor_en(self, dias: np.ndarray) -> np.ndarray:
        """
        Factor de descuento (S, F) para cada plazo residual en `dias` (F,). Interpola linealmente el
        factor entre los dos plazos de la curva que encierran el flujo. Plazo 0 → 0 (flujo vencido).
        """
        dias = np.asarray(dias, dtype=float)
        out = np.zeros((len(self.fechas), len(dias)))
        p = self.plazos
        for j, d in enumerate(dias):
            if d <= 0:
                continue
            if d <= p[0]:
                out[:, j] = 1.0 / (1.0 + self.tasas[:, 0] / 100.0) ** (d / BASE_DIAS)
            elif d >= p[-1]:
                out[:, j] = 1.0 / (1.0 + self.tasas[:, -1] / 100.0) ** (d / BASE_DIAS)
            else:
                k = int(np.searchsorted(p, d))          # p[k-1] < d <= p[k]
                if p[k] == d:
                    out[:, j] = self.factores[:, k]
                else:
                    w = (d - p[k - 1]) / (p[k] - p[k - 1])
                    out[:, j] = self.factores[:, k - 1] + (self.factores[:, k] - self.factores[:, k - 1]) * w
        return out


def construir_escenarios_curva(
    puntos: List[dict], fecha_portafolio: pd.Timestamp, desde: pd.Timestamp, hasta: pd.Timestamp, nombre: str
) -> EscenariosCurva:
    if not puntos:
        raise ErrorDatosRentaFija(f"La curva '{nombre}' no trae puntos.")
    df = pd.DataFrame(puntos)
    faltan = {"fecha", "plazoDias", "tasa"} - set(df.columns)
    if faltan:
        raise ErrorDatosRentaFija(f"Curva '{nombre}': faltan campos {sorted(faltan)}.")
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.normalize()
    df["tasa"] = pd.to_numeric(df["tasa"], errors="coerce")
    df["plazoDias"] = pd.to_numeric(df["plazoDias"], errors="coerce")
    df = df.dropna(subset=["tasa", "plazoDias"])

    # Matriz fecha x plazo; una curva incompleta en un día se rellena con el día anterior.
    m = df.pivot_table(index="fecha", columns="plazoDias", values="tasa", aggfunc="last").sort_index()
    m = m.ffill()
    if fecha_portafolio not in m.index:
        raise ErrorDatosRentaFija(
            f"La curva '{nombre}' no tiene datos en la fecha del portafolio ({fecha_portafolio.date()})."
        )
    m = m.loc[:, m.loc[fecha_portafolio].notna()]      # plazos sin dato en la fecha base no sirven
    if m.shape[1] == 0:
        raise ErrorDatosRentaFija(f"La curva '{nombre}' no tiene plazos en la fecha del portafolio.")
    variacion = m.diff()
    base = m.loc[fecha_portafolio]
    esc = (variacion + base).clip(lower=0.0)
    esc.loc[fecha_portafolio] = base.clip(lower=0.0)       # sin perturbación en la fecha base

    esc = esc.loc[(esc.index >= desde) & (esc.index <= hasta)].dropna(how="any")
    if fecha_portafolio not in esc.index:
        raise ErrorDatosRentaFija(f"La ventana de la curva '{nombre}' no incluye la fecha del portafolio.")
    if len(esc) < 3:
        raise ErrorDatosRentaFija(f"La curva '{nombre}' tiene solo {len(esc)} escenarios en la ventana.")

    plazos = esc.columns.to_numpy(dtype=float)
    tasas = esc.to_numpy(dtype=float)
    factores = 1.0 / (1.0 + tasas / 100.0) ** (plazos[None, :] / BASE_DIAS)
    fechas = pd.DatetimeIndex(esc.index)
    return EscenariosCurva(plazos, fechas, tasas, factores, int(fechas.get_loc(fecha_portafolio)))


def construir_escenarios_tc(
    puntos: List[dict], fecha_portafolio: pd.Timestamp, desde: pd.Timestamp, hasta: pd.Timestamp, par: str
) -> pd.Series:
    """Serie de TC simulado (indexada por fecha) = TC_base * exp(retorno log diario)."""
    df = pd.DataFrame(puntos)
    if df.empty or not {"fecha", "tc"} <= set(df.columns):
        raise ErrorDatosRentaFija(f"Tipo de cambio '{par}': se espera una lista de {{fecha, tc}}.")
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.normalize()
    s = pd.to_numeric(df.set_index("fecha")["tc"], errors="coerce").dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if fecha_portafolio not in s.index:
        raise ErrorDatosRentaFija(f"El tipo de cambio '{par}' no tiene dato en {fecha_portafolio.date()}.")
    tc_base = float(s.loc[fecha_portafolio])
    ret = np.log(s / s.shift(1))
    ret.loc[fecha_portafolio] = 0.0
    sim = tc_base * np.exp(ret)
    return sim.loc[(sim.index >= desde) & (sim.index <= hasta)].dropna()


# ======================================================================
# Valorización por instrumento
# ======================================================================
def _flujos_df(instr: dict, fecha_portafolio: pd.Timestamp) -> pd.DataFrame:
    flujos = instr.get("flujos") or []
    tipo = instr["tipo"]
    if tipo == "CD" and not flujos:
        venc = instr.get("fechaVencimiento")
        if venc is None:
            raise ErrorDatosRentaFija(f"CD {instr['isin']}: indique 'fechaVencimiento' o un flujo.")
        flujos = [{"fecha": venc, "montoPct": 100.0}]
    if not flujos:
        raise ErrorDatosRentaFija(f"Bono {instr['isin']}: no tiene flujos.")
    df = pd.DataFrame(flujos)
    if not {"fecha", "montoPct"} <= set(df.columns):
        raise ErrorDatosRentaFija(f"{instr['isin']}: cada flujo necesita 'fecha' y 'montoPct'.")
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.normalize()
    df["montoPct"] = pd.to_numeric(df["montoPct"], errors="coerce").fillna(0.0)
    df["dias"] = (df["fecha"] - fecha_portafolio).dt.days.clip(lower=0)
    for c in ("diasCorrido", "diasPlazo"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0) if c in df.columns else 0.0
    return df.sort_values("fecha").reset_index(drop=True)


def _tir(flujos_monto: np.ndarray, t_anios: np.ndarray, precio: float) -> Optional[float]:
    """Tasa efectiva anual tal que VP(flujos) = precio (bisección; act/360)."""
    mask = t_anios > 0
    cf, t = flujos_monto[mask], t_anios[mask]
    if cf.size == 0 or precio <= 0:
        return None
    f = lambda y: float(np.sum(cf / (1.0 + y) ** t) - precio)   # noqa: E731
    lo, hi = -0.99, 10.0
    if f(lo) * f(hi) > 0:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


class MotorRentaFija:
    def __init__(self, parametros: dict, instrumentos: List[dict], curvas: Dict[str, List[dict]],
                 tipos_cambio: Optional[Dict[str, List[dict]]] = None):
        if not instrumentos:
            raise ErrorDatosRentaFija("La lista 'instrumentos' está vacía.")
        self.moneda_funcional = parametros.get("monedaFuncional", "PEN")
        self.fecha_portafolio = _fecha(parametros["fechaPortafolio"], "fechaPortafolio")
        self.desde = _fecha(parametros["fechaDesde"], "fechaDesde")
        self.hasta = _fecha(parametros.get("fechaHasta", parametros["fechaPortafolio"]), "fechaHasta")
        if self.desde >= self.fecha_portafolio:
            raise ErrorDatosRentaFija("'fechaDesde' debe ser anterior a la fecha del portafolio.")
        self.niveles = parametros.get("nivelesConfianza") or [0.99]
        for n in self.niveles:
            if not 0 < n < 1:
                raise ErrorDatosRentaFija(f"Nivel de confianza fuera de (0,1): {n}")

        self.instrumentos = instrumentos
        self._curvas_raw = curvas or {}
        self._tc_raw = tipos_cambio or {}
        self._esc_curva: Dict[str, EscenariosCurva] = {}
        self._esc_tc: Dict[str, pd.Series] = {}

    # --- caches de escenarios ---
    def _curva(self, codigo: str) -> EscenariosCurva:
        if codigo not in self._esc_curva:
            if codigo not in self._curvas_raw:
                raise ErrorDatosRentaFija(f"Falta la curva '{codigo}' en 'curvas'.")
            self._esc_curva[codigo] = construir_escenarios_curva(
                self._curvas_raw[codigo], self.fecha_portafolio, self.desde, self.hasta, codigo)
        return self._esc_curva[codigo]

    def _tc(self, moneda: str) -> Optional[pd.Series]:
        if moneda == self.moneda_funcional:
            return None
        par = f"{moneda}/{self.moneda_funcional}"
        if par not in self._esc_tc:
            if par not in self._tc_raw:
                raise ErrorDatosRentaFija(f"Falta el tipo de cambio '{par}' en 'tiposCambio'.")
            self._esc_tc[par] = construir_escenarios_tc(
                self._tc_raw[par], self.fecha_portafolio, self.desde, self.hasta, par)
        return self._esc_tc[par]

    # --- valorización de un instrumento ---
    def _valorizar(self, instr: dict) -> dict:
        for c in ("isin", "tipo", "moneda", "curva", "nominal"):
            if c not in instr:
                raise ErrorDatosRentaFija(f"Instrumento sin '{c}': {instr}")
        if instr["tipo"] not in TIPOS_VALIDOS:
            raise ErrorDatosRentaFija(f"{instr['isin']}: tipo '{instr['tipo']}' no soportado ({TIPOS_VALIDOS}).")

        curva = self._curva(instr["curva"])
        flujos = _flujos_df(instr, self.fecha_portafolio)
        nominal = float(instr["nominal"])
        if (flujos["dias"] > 0).sum() == 0:
            raise ErrorDatosRentaFija(f"{instr['isin']} está vencido al {self.fecha_portafolio.date()}.")

        # VP en moneda de origen por escenario = nominal * sum(monto% / 100 * factor)
        factores = curva.factor_en(flujos["dias"].to_numpy())                  # (S, F)
        vp = nominal * (factores * (flujos["montoPct"].to_numpy() / 100.0)[None, :]).sum(axis=1)
        serie = pd.Series(vp, index=curva.fechas, name=instr["isin"])

        tc = self._tc(instr["moneda"])
        if tc is None:
            vp_fun = serie
        else:
            comunes = serie.index.intersection(tc.index)
            if self.fecha_portafolio not in comunes:
                raise ErrorDatosRentaFija(f"{instr['isin']}: sin TC en la fecha del portafolio.")
            vp_fun = serie.loc[comunes] * tc.loc[comunes]

        base = float(vp_fun.loc[self.fecha_portafolio])
        pnl = vp_fun - base
        riesgo = self._riesgo(instr, flujos, curva, nominal, float(serie.loc[self.fecha_portafolio]))
        return {"isin": instr["isin"], "tipo": instr["tipo"], "moneda": instr["moneda"], "curva": instr["curva"],
                "mtm": base, "mtmOrigen": float(serie.loc[self.fecha_portafolio]), "pnl": pnl, **riesgo}

    def _riesgo(self, instr, flujos, curva: EscenariosCurva, nominal: float, mtm_origen: float) -> dict:
        """Duración, convexidad, TIR e interés corrido en moneda de origen, con la curva base."""
        vivos = flujos[flujos["dias"] > 0]
        t = vivos["dias"].to_numpy() / BASE_DIAS
        cf = nominal * vivos["montoPct"].to_numpy() / 100.0
        f_base = curva.factor_en(vivos["dias"].to_numpy())[curva.idx_base]
        pv = cf * f_base
        total = pv.sum()
        macaulay = float((pv * t).sum() / total) if total else 0.0
        tir = _tir(cf, t, mtm_origen)
        if tir is not None:
            modificada = macaulay / (1.0 + tir)
            convexidad = float((pv * t * (t + 1.0)).sum() / total / (1.0 + tir) ** 2) if total else 0.0
        else:
            modificada, convexidad = None, None
        corrido = float(np.sum(
            np.where(flujos["diasPlazo"] > 0, flujos["diasCorrido"] / flujos["diasPlazo"].replace(0, np.nan), 0.0)
            * flujos["montoPct"] / 100.0 * nominal))
        return {"duracionMacaulay": macaulay, "duracionModificada": modificada, "convexidad": convexidad,
                "tir": tir, "interesCorrido": corrido, "precioLimpio": (mtm_origen - corrido) / nominal}

    # --- cálculo total ---
    def calcular(self) -> dict:
        inicio = time.time()
        val = [self._valorizar(i) for i in self.instrumentos]

        pnl_df = pd.concat([v["pnl"].rename(v["isin"]) for v in val], axis=1, join="inner").sort_index()
        if self.fecha_portafolio not in pnl_df.index or len(pnl_df) < 3:
            raise ErrorDatosRentaFija("No hay escenarios comunes a todos los instrumentos en la ventana pedida.")
        pnl_total = pnl_df.sum(axis=1)
        mtm_total = float(sum(v["mtm"] for v in val))

        resultados = []
        for nivel in self.niveles:
            var, cvar = self._var_cvar(pnl_total.to_numpy(), nivel)
            por_instr = {}
            for col in pnl_df.columns:
                vi, ci = self._var_cvar(pnl_df[col].to_numpy(), nivel)
                por_instr[col] = {"var": vi, "cvar": ci}
            resultados.append({
                "nivelConfianza": nivel, "var": var, "cvar": cvar,
                "varNoDiversificado": float(sum(x["var"] for x in por_instr.values())),
                "varIndividual": {k: x["var"] for k, x in por_instr.items()},
                "cvarIndividual": {k: x["cvar"] for k, x in por_instr.items()},
            })
            resultados[-1]["beneficioDiversificacion"] = resultados[-1]["varNoDiversificado"] - var

        pesos = {v["isin"]: (v["mtm"] / mtm_total if mtm_total else 0.0) for v in val}
        dur_mac = sum(pesos[v["isin"]] * v["duracionMacaulay"] for v in val)
        dur_mod = sum(pesos[v["isin"]] * (v["duracionModificada"] or 0.0) for v in val)
        conv = sum(pesos[v["isin"]] * (v["convexidad"] or 0.0) for v in val)

        return {
            "fechaPortafolio": self.fecha_portafolio.strftime("%Y-%m-%d"),
            "monedaReporte": self.moneda_funcional,
            "mtmTotal": mtm_total,
            "numEscenarios": int(len(pnl_df)),
            "ventana": {"desde": pnl_df.index.min().strftime("%Y-%m-%d"), "hasta": pnl_df.index.max().strftime("%Y-%m-%d")},
            "resultados": resultados,
            "riesgoCartera": {"duracionMacaulay": dur_mac, "duracionModificada": dur_mod, "convexidad": conv},
            "instrumentos": [
                {k: v[k] for k in ("isin", "tipo", "moneda", "curva", "mtm", "mtmOrigen", "duracionMacaulay",
                                   "duracionModificada", "convexidad", "tir", "interesCorrido", "precioLimpio")}
                | {"peso": pesos[v["isin"]]}
                for v in val
            ],
            "distribucionHistorica": pnl_total.to_numpy().tolist(),
            "fechasEscenarios": [d.strftime("%Y-%m-%d") for d in pnl_df.index],
            "tiempoCalculoMs": (time.time() - inicio) * 1000,
        }

    @staticmethod
    def _var_cvar(pnl: np.ndarray, confianza: float):
        """Percentil (1-conf) del P&L; CVaR = promedio de los P&L <= VaR. Ambos en positivo (monto en riesgo)."""
        pnl = pnl[np.isfinite(pnl)]
        if pnl.size == 0:
            return 0.0, 0.0
        var = float(np.percentile(pnl, (1.0 - confianza) * 100.0))
        cola = pnl[pnl <= var]
        cvar = float(cola.mean()) if cola.size else var
        return abs(var), abs(cvar)
