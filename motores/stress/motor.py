"""
motores/stress/motor.py
========================
Motor de Stress Testing: revaloriza el portafolio ACTUAL bajo un escenario de shock (variación de
precio y/o de tipo de cambio) e informa el impacto real en el valor de mercado.

A diferencia del VaR (que estima una pérdida probable a partir del historial), un stress test parte
de un supuesto explícito del usuario ("¿qué pasa si el mercado cae 20%?") y hace una revaluación
EXACTA del portafolio bajo ese supuesto: no hay nada simulado ni aproximado en el cálculo en sí,
solo en la magnitud del shock, que es información de entrada, no una salida del motor.
"""

from dataclasses import dataclass
from typing import Dict, List


class ErrorDatosStress(Exception):
    """Error de datos de entrada (activos vacíos, MTM no positivo, etc.), no un bug del motor."""


@dataclass
class ImpactoActivo:
    nombre: str
    mtmActual: float
    mtmEstresado: float
    impacto: float

    def to_dict(self) -> dict:
        return {
            "nombre": self.nombre,
            "mtmActual": self.mtmActual,
            "mtmEstresado": self.mtmEstresado,
            "impacto": self.impacto,
        }


class MotorStress:
    """Recibe las posiciones actuales (mismo formato 'activos' que el motor de VaR) y un escenario
    de shock, y devuelve el valor de mercado actual vs. estresado, total y por activo."""

    def __init__(self, parametros: dict, activos: List[dict]):
        if not activos:
            raise ErrorDatosStress("La lista 'activos' está vacía.")
        self.moneda_reporte = parametros["monedaReporte"]
        self.tc_actual: Dict[str, float] = parametros.get("tcActual") or {}
        self.activos = activos

    def calcular(self, escenario: dict) -> dict:
        shock_precio_global = float(escenario.get("shockPrecioPct", 0.0) or 0.0)
        shock_fx = float(escenario.get("shockCambiarioPct", 0.0) or 0.0)
        shocks_por_activo: Dict[str, float] = escenario.get("shockPrecioPorActivo") or {}

        impactos: List[ImpactoActivo] = []
        mtm_actual_total = 0.0
        mtm_estresado_total = 0.0

        for activo in self.activos:
            nombre = activo["nombre"]
            cantidad = float(activo["numAcciones"])
            precio = float(activo["precioActual"])
            moneda_activo = activo["monedaActivo"]

            es_moneda_extranjera = moneda_activo != self.moneda_reporte
            tc_actual = self.tc_actual.get(f"{moneda_activo}_{self.moneda_reporte}", 1.0) if es_moneda_extranjera else 1.0
            tc_estresado = tc_actual * (1 + shock_fx) if es_moneda_extranjera else 1.0

            shock_especifico = shocks_por_activo.get(nombre, shock_precio_global)
            precio_estresado = precio * (1 + shock_especifico)

            mtm_actual = cantidad * precio * tc_actual
            mtm_estresado = cantidad * precio_estresado * tc_estresado

            impactos.append(ImpactoActivo(nombre, mtm_actual, mtm_estresado, mtm_estresado - mtm_actual))
            mtm_actual_total += mtm_actual
            mtm_estresado_total += mtm_estresado

        if mtm_actual_total <= 0:
            raise ErrorDatosStress("El valor de mercado (MTM) actual del portafolio debe ser positivo.")

        impacto_total = mtm_estresado_total - mtm_actual_total
        return {
            "mtmActual": mtm_actual_total,
            "mtmEstresado": mtm_estresado_total,
            "impactoTotal": impacto_total,
            "impactoPct": impacto_total / mtm_actual_total,
            "porActivo": [i.to_dict() for i in impactos],
        }
