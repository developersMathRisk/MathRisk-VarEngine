"""Pruebas unitarias del motor de renta fija con datos sintéticos (sin archivos externos)."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from motores.renta_fija.motor import ErrorDatosRentaFija, MotorRentaFija  # noqa: E402
from app import app  # noqa: E402

PLAZOS = [90, 360, 720, 1800]


def curva(tasa0=4.0, dias=40, paso=0.01):
    """Curva plana que sube `paso` puntos por día desde 2020-01-01 (variación diaria constante)."""
    fechas = pd.bdate_range("2020-01-01", periods=dias)
    return [{"fecha": f.strftime("%Y-%m-%d"), "plazoDias": p, "tasa": tasa0 + paso * i}
            for i, f in enumerate(fechas) for p in PLAZOS], fechas


def tc(fechas, base=3.5, paso=0.01):
    return [{"fecha": f.strftime("%Y-%m-%d"), "tc": base + paso * i} for i, f in enumerate(fechas)]


class TestRentaFija(unittest.TestCase):
    def setUp(self):
        self.cur, self.fechas = curva()
        self.hoy = self.fechas[-1].strftime("%Y-%m-%d")
        self.par = {"fechaPortafolio": self.hoy, "fechaDesde": "2020-01-02", "nivelesConfianza": [0.95]}

    def _motor(self, instr, **kw):
        return MotorRentaFija(self.par, instr, {"C": self.cur}, kw.get("tc"))

    def test_cd_mtm_es_valor_presente_con_factor_de_curva(self):
        cd = {"isin": "CD1", "tipo": "CD", "moneda": "PEN", "curva": "C", "nominal": 1_000_000,
              "fechaVencimiento": (self.fechas[-1] + pd.Timedelta(days=360)).strftime("%Y-%m-%d")}
        r = self._motor([cd]).calcular()
        tasa_hoy = 4.0 + 0.01 * (len(self.fechas) - 1)
        esperado = 1_000_000 / (1 + tasa_hoy / 100) ** (360 / 360)
        self.assertAlmostEqual(r["mtmTotal"], esperado, places=4)

    def test_curva_creciente_da_perdidas_en_renta_fija(self):
        """Con tasas que solo suben el P&L de un bono a tasa fija es <= 0 salvo la fecha base."""
        bono = {"isin": "B1", "tipo": "BONO", "moneda": "PEN", "curva": "C", "nominal": 100,
                "flujos": [{"fecha": (self.fechas[-1] + pd.Timedelta(days=d)).strftime("%Y-%m-%d"), "montoPct": m}
                           for d, m in ((180, 3.0), (360, 3.0), (720, 103.0))]}
        r = self._motor([bono]).calcular()
        self.assertGreater(r["resultados"][0]["var"], 0)
        dist = np.array(r["distribucionHistorica"])
        self.assertAlmostEqual(dist[-1], 0.0, places=9)           # la fecha base no se perturba

    def test_efecto_tipo_de_cambio_en_usd(self):
        cd = {"isin": "CDU", "tipo": "CD", "moneda": "USD", "curva": "C", "nominal": 1000,
              "fechaVencimiento": (self.fechas[-1] + pd.Timedelta(days=360)).strftime("%Y-%m-%d")}
        r = self._motor([cd], tc={"USD/PEN": tc(self.fechas)}).calcular()
        tasa_hoy = 4.0 + 0.01 * (len(self.fechas) - 1)
        self.assertAlmostEqual(r["mtmTotal"], 1000 / (1 + tasa_hoy / 100) * (3.5 + 0.01 * (len(self.fechas) - 1)), places=4)

    def test_tc_se_une_por_fecha_no_por_posicion(self):
        """Si al TC le falta un día intermedio, ese escenario se descarta; no se desplaza el resto."""
        cd = {"isin": "CDU", "tipo": "CD", "moneda": "USD", "curva": "C", "nominal": 1000,
              "fechaVencimiento": (self.fechas[-1] + pd.Timedelta(days=360)).strftime("%Y-%m-%d")}
        serie = tc(self.fechas)
        del serie[10]
        r = self._motor([cd], tc={"USD/PEN": serie}).calcular()
        self.assertNotIn(self.fechas[10].strftime("%Y-%m-%d"), r["fechasEscenarios"])

    def test_var_cvar_percentil(self):
        var, cvar = MotorRentaFija._var_cvar(np.arange(-100, 101, dtype=float), 0.95)
        self.assertAlmostEqual(var, 90.0)
        self.assertGreaterEqual(cvar, var)

    def test_bono_vencido_da_error(self):
        bono = {"isin": "B0", "tipo": "BONO", "moneda": "PEN", "curva": "C", "nominal": 100,
                "flujos": [{"fecha": "2019-01-01", "montoPct": 100}]}
        with self.assertRaises(ErrorDatosRentaFija):
            self._motor([bono]).calcular()

    def test_falta_curva_o_tc(self):
        cd = {"isin": "CDU", "tipo": "CD", "moneda": "USD", "curva": "C", "nominal": 1,
              "fechaVencimiento": "2021-06-30"}
        with self.assertRaises(ErrorDatosRentaFija):
            self._motor([cd]).calcular()                       # sin tiposCambio
        with self.assertRaises(ErrorDatosRentaFija):
            MotorRentaFija(self.par, [dict(cd, curva="X", moneda="PEN")], {"C": self.cur}).calcular()

    def test_endpoint(self):
        cliente = app.test_client()
        cd = {"isin": "CD1", "tipo": "CD", "moneda": "PEN", "curva": "C", "nominal": 1_000_000,
              "fechaVencimiento": "2021-06-30"}
        ok = cliente.post("/renta-fija/calcular", json={"parametros": self.par, "instrumentos": [cd], "curvas": {"C": self.cur}})
        self.assertEqual(ok.status_code, 200, ok.get_data(as_text=True))
        self.assertIn("mtmTotal", ok.get_json())
        self.assertEqual(cliente.post("/renta-fija/calcular", json={"parametros": {}}).status_code, 400)
        self.assertEqual(cliente.post("/renta-fija/calcular", data="x").status_code, 400)
        self.assertIn("renta-fija", cliente.get("/salud").get_json()["motoresDisponibles"])


if __name__ == "__main__":
    unittest.main()
