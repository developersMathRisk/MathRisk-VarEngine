"""Pruebas del motor de VaR unificado: acciones/fondos + renta fija en un mismo cálculo."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from motores.var.motor import ErrorDatosVaR, MotorVaR  # noqa: E402
from app import app  # noqa: E402

PLAZOS = [90, 360, 720, 1800]
N = 60


def fechas():
    return pd.bdate_range("2024-01-01", periods=N)


def escenarios(fs, precios, tc=None):
    """Escenarios en el formato del backend: numero mayor = más antiguo."""
    out = []
    for i, f in enumerate(fs):
        e = {"numero": len(fs) - i, "fecha": f.strftime("%Y-%m-%d"), "precios": {"ACC": float(precios[i])}}
        if tc is not None:
            e["tcHistorico"] = {"USD_PEN": float(tc[i])}
        out.append(e)
    return out


def curva(fs, tasas):
    return [{"fecha": f.strftime("%Y-%m-%d"), "plazoDias": p, "tasa": float(t)}
            for f, t in zip(fs, tasas) for p in PLAZOS]


def bloque_rf(fs, tasas, moneda="PEN", nominal=1_000_000):
    hoy = fs[-1]
    return {
        "fechaValoracion": hoy.strftime("%Y-%m-%d"),
        "curvas": {"C": curva(fs, tasas)},
        "instrumentos": [{
            "isin": "BONO1", "moneda": moneda, "curva": "C", "nominal": nominal,
            "flujos": [{"fecha": (hoy + pd.Timedelta(days=d)).strftime("%Y-%m-%d"), "montoPct": m}
                       for d, m in ((180, 3.0), (360, 3.0), (720, 103.0))],
        }],
    }


PARAMS = {"monedaReporte": "PEN", "metodologias": ["historico", "montecarlo", "parametrico"],
          "nivelesConfianza": [0.95, 0.99], "tcActual": {}}


class TestVarIntegrado(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.fs = fechas()
        self.precios = 100 * np.cumprod(1 + rng.normal(0, 0.01, N))
        self.tasas = 5 + np.cumsum(rng.normal(0, 0.03, N))
        self.activos = [{"nombre": "ACC", "numAcciones": 1000, "precioActual": float(self.precios[-1]), "monedaActivo": "PEN"}]

    def calcular(self, activos, rf):
        return MotorVaR(PARAMS, activos, escenarios(self.fs, self.precios), rf).calcular(
            PARAMS["metodologias"], PARAMS["nivelesConfianza"], 5000)

    def test_sin_renta_fija_no_cambia_el_resultado(self):
        r1 = MotorVaR(PARAMS, self.activos, escenarios(self.fs, self.precios)).calcular(["historico"], [0.99])
        r2 = self.calcular(self.activos, None)
        h2 = next(x for x in r2["resultados"] if x["metodologia"] == "historico" and x["nivelConfianza"] == 0.99)
        self.assertAlmostEqual(r1["resultados"][0]["varDiversificado"], h2["varDiversificado"], places=9)
        self.assertNotIn("rentaFija", r2)

    def test_solo_bonos_revalua_con_cambios_de_curva(self):
        r = self.calcular([], bloque_rf(self.fs, self.tasas))
        self.assertEqual(r["numEscenarios"], N - 1)
        self.assertIn("BONO1", r["mtmPorActivo"])
        # El P&L del escenario k es la revaluación con base + (tasa[k] - tasa[k-1]): verificado a mano para k=1
        rf = r["rentaFija"]["instrumentos"][0]
        self.assertGreater(rf["duracionModificada"], 0)
        for x in r["resultados"]:
            self.assertLess(x["varDiversificado"], 0)
            self.assertLessEqual(x["cvarDiversificado"], x["varDiversificado"] + 1e-9)

    def test_portafolio_mixto_diversifica(self):
        r = self.calcular(self.activos, bloque_rf(self.fs, self.tasas))
        self.assertEqual(set(r["mtmPorActivo"]), {"ACC", "BONO1"})
        for x in r["resultados"]:
            # VaR diversificado nunca peor que la suma de los individuales (subaditividad en estos datos)
            self.assertGreaterEqual(x["varDiversificado"], x["varNoDiversificado"] - 1e-6)
            self.assertEqual(set(x["varIndividual"]), {"ACC", "BONO1"})

    def test_bono_en_dolares_usa_tc_del_escenario(self):
        tc = 3.7 + np.cumsum(np.full(N, 0.002))
        par = {**PARAMS, "tcActual": {"USD_PEN": float(tc[-1])}}
        m = MotorVaR(par, [], escenarios(self.fs, self.precios, tc), bloque_rf(self.fs, np.full(N, 5.0), "USD", 1000))
        r = m.calcular(["historico"], [0.99])
        rf = r["rentaFija"]["instrumentos"][0]
        self.assertAlmostEqual(rf["mtm"], rf["mtmOrigen"] * tc[-1], places=6)
        # Curva constante: toda la variación viene del tipo de cambio
        esperado = rf["mtm"] * (tc[1:] / tc[:-1] - 1)
        np.testing.assert_allclose(r["distribucionHistorica"], esperado, rtol=1e-9)

    def test_curva_que_no_cubre_los_escenarios_falla_claro(self):
        rf = bloque_rf(self.fs, self.tasas)
        rf["curvas"]["C"] = [p for p in rf["curvas"]["C"] if p["fecha"] >= self.fs[10].strftime("%Y-%m-%d")]
        with self.assertRaises(ErrorDatosVaR) as ctx:
            self.calcular(self.activos, rf)
        self.assertIn("no cubre", str(ctx.exception))

    def test_endpoint_acepta_renta_fija_sin_activos(self):
        cuerpo = {"parametros": PARAMS, "activos": [], "escenarios": escenarios(self.fs, self.precios),
                  "rentaFija": bloque_rf(self.fs, self.tasas)}
        with app.test_client() as c:
            os.environ.pop("MOTOR_API_KEY", None)
            resp = c.post("/var/calcular", json=cuerpo)
        self.assertEqual(resp.status_code, 200, resp.get_json())
        self.assertIn("rentaFija", resp.get_json())


if __name__ == "__main__":
    unittest.main()
