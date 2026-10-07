"""
Regresión del motor de renta fija contra el original (PrometheusModelos / VaR - 190926).
Uso:  python tests/regresion_renta_fija.py "<ruta a 'VaR - 190926'>"
Corre las funciones originales (valorización por escenario) sobre Inputs/ + cache_temporal/ y compara el
MTM y la serie de P&L por instrumento con el motor portado. No es parte de la suite unitaria: depende de
archivos externos.
"""
import json
import os
import sys
import types
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from motores.renta_fija.motor import MotorRentaFija  # noqa: E402

if len(sys.argv) < 2:
    sys.exit(__doc__)
RAIZ = sys.argv[1]
sys.modules.setdefault("numpy_financial", types.SimpleNamespace(irr=lambda x: float("nan")))  # solo se usa en el bloque de riesgo
sys.path.insert(0, RAIZ)
import HerramientaVaROptimizado as H  # noqa: E402

FECHA = datetime(2020, 6, 19)
DESDE = datetime(2019, 7, 1)
INP = os.path.join(RAIZ, "Inputs")


def curva_original(codigo):
    if codigo == "CBCRS":
        df = pd.read_csv(os.path.join(INP, "CurvasCD.csv"), sep=";", decimal=",")
        df.columns = ["fecha_proceso", "tipo_curva", "plazo_dias", "tasa"]
        df["fecha_proceso"] = pd.to_datetime(df["fecha_proceso"], dayfirst=True)
        return df
    partes = [pd.read_parquet(os.path.join(RAIZ, "cache_temporal", f"{codigo}_{a}.parquet")) for a in (2019, 2020)]
    df = pd.concat(partes)
    df.columns = ["sec", "fecha_proceso", "plazo_dias", "tasa"]
    df["tipo_curva"] = codigo
    return df[["fecha_proceso", "tipo_curva", "plazo_dias", "tasa"]]


def tc_original():
    df = pd.read_csv(os.path.join(INP, "HistoricoTCProm.csv"), sep=";", usecols=[0, 1, 2], decimal=",")
    df.columns = ["fecha_proceso", "moneda", "tipoCambio"]
    df["fecha_proceso"] = pd.to_datetime(df["fecha_proceso"], dayfirst=True)
    df["moneda"] = "USD/PEN"
    return df


def main():
    port = pd.read_csv(os.path.join(INP, "Portafolio.csv"), sep=";")
    port = port[port["TipoInstrumento"].isin(["CD", "BONO"]) & (port["CodCarteraCtble"] == "PRUEBA")].reset_index(drop=True)
    flujo = pd.read_csv(os.path.join(INP, "Flujo.csv"), sep=";")
    flujo = flujo[flujo["CodISIN"].isin(port["CodISIN"])]

    curvas = {c: curva_original(c) for c in port["CodCurvaMTM"].unique()}
    tc = tc_original()

    # ---------- original ----------
    orig = {}
    tc_fin = H.calcular_FactorTC(tc, FECHA, DESDE)
    for _, r in port.iterrows():
        cf = H.calcular_FactorCurva(curvas[r["CodCurvaMTM"]], FECHA, DESDE)
        base = cf[cf["fecha_proceso"] == FECHA].copy()
        fl = flujo[flujo["CodISIN"] == r["CodISIN"]]
        fvcto = datetime.strptime(r["FechaVcto"], "%d/%m/%Y %H:%M")
        if fl.empty:
            fl = pd.DataFrame([{"CodISIN": r["CodISIN"], "fechaVcto": fvcto, "nominal": r["MtoNominal"], "plazo": (fvcto - FECHA).days}])
        det = pd.DataFrame([{"CodISIN": r["CodISIN"], "Val_Nominal": float(r["MtoNominal"]), "Val_Flujos": fl.to_json(orient="records"),
                             "Val_FechaMTM": "19/06/2020", "CodCurvaMTM": r["CodCurvaMTM"]}])
        f = H.ValorizarInstrumentoCD if r["TipoInstrumento"] == "CD" else H.ValorizarInstrumentoBono
        res = f(FECHA, det, base, cf)
        tcf = tc_fin if r["CodMoneda"] == "USD" else None
        out = H.EfectoTasaTC(res, tcf, FECHA)
        orig[r["CodISIN"]] = out.set_index("fecha_proceso")

    # ---------- portado ----------
    instr, cur = [], {}
    for _, r in port.iterrows():
        d = {"isin": r["CodISIN"], "tipo": r["TipoInstrumento"], "moneda": r["CodMoneda"], "curva": r["CodCurvaMTM"], "nominal": float(r["MtoNominal"])}
        if r["TipoInstrumento"] == "CD":
            d["fechaVencimiento"] = pd.to_datetime(r["FechaVcto"], dayfirst=True).strftime("%Y-%m-%d")
        else:
            fl = flujo[flujo["CodISIN"] == r["CodISIN"]]
            d["flujos"] = [{"fecha": pd.to_datetime(x.FecVcto, dayfirst=True).strftime("%Y-%m-%d"), "montoPct": float(x.MtoFlujoBono),
                            "diasCorrido": float(x.DiasCorrido), "diasPlazo": float(x.DiasPlazo)} for x in fl.itertuples()]
        instr.append(d)
    for c, df in curvas.items():
        cur[c] = [{"fecha": a.strftime("%Y-%m-%d"), "plazoDias": int(p), "tasa": float(t)} for a, p, t in zip(df.fecha_proceso, df.plazo_dias, df.tasa)]
    tcs = {"USD/PEN": [{"fecha": a.strftime("%Y-%m-%d"), "tc": float(t)} for a, t in zip(tc.fecha_proceso, tc.tipoCambio)]}
    motor = MotorRentaFija({"fechaPortafolio": "2020-06-19", "fechaDesde": "2019-07-01", "nivelesConfianza": [0.95, 0.99]}, instr, cur, tcs)
    res = motor.calcular()

    # ---------- comparación ----------
    print(f"{'ISIN':<14}{'MTM orig':>18}{'MTM nuevo':>18}{'dMTM':>12}{'max|dPnL|':>14}{'escen.':>8}")
    peor = 0.0
    for i in res["instrumentos"]:
        o = orig[i["isin"]]
        mtm_o = float(o.loc[FECHA, "VPNominalPEN"])
        pn = motor._valorizar(next(x for x in instr if x["isin"] == i["isin"]))["pnl"]
        comun = o.index.intersection(pn.index)
        d = float(np.max(np.abs(o.loc[comun, "PnLSimulado"].to_numpy() - pn.loc[comun].to_numpy())))
        peor = max(peor, d)
        print(f"{i['isin']:<14}{mtm_o:>18,.2f}{i['mtm']:>18,.2f}{i['mtm']-mtm_o:>12.4f}{d:>14.6f}{len(comun):>8}")
    print("\nMTM total:", f"{res['mtmTotal']:,.2f}", "| escenarios:", res["numEscenarios"], "| ventana:", res["ventana"])
    for r in res["resultados"]:
        print(f"VaR {r['nivelConfianza']:.0%}: {r['var']:,.2f}  CVaR: {r['cvar']:,.2f}  (no diversificado {r['varNoDiversificado']:,.2f})")
    print("riesgo cartera:", {k: round(v, 4) for k, v in res["riesgoCartera"].items()})
    print("max |dPnL| global:", peor)


if __name__ == "__main__":
    main()
