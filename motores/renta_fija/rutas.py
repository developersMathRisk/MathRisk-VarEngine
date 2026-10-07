"""
motores/renta_fija/rutas.py
=============================
Endpoints HTTP del motor de renta fija (CD y bonos), como Blueprint independiente. Registrado en
app.py bajo el prefijo "/renta-fija".

Contrato de POST /renta-fija/calcular:
{
  "parametros": {"fechaPortafolio": "2020-06-19", "fechaDesde": "2019-07-01", "fechaHasta": "2020-06-19",
                 "monedaFuncional": "PEN", "nivelesConfianza": [0.95, 0.99]},
  "instrumentos": [{"isin": "...", "tipo": "CD"|"BONO", "moneda": "PEN", "curva": "CBCRS", "nominal": 1500000,
                    "fechaVencimiento": "2021-02-02",                     (CD; o un flujo con montoPct 100)
                    "flujos": [{"fecha": "2021-02-02", "montoPct": 4.375, "diasCorrido": 0, "diasPlazo": 180}]}],
  "curvas": {"CBCRS": [{"fecha": "2020-06-19", "plazoDias": 90, "tasa": 3.03}, ...]},
  "tiposCambio": {"USD/PEN": [{"fecha": "2020-06-19", "tc": 3.45}, ...]}
}
"""

from datetime import datetime, timezone

from flask import Blueprint, current_app, jsonify, request

from .motor import ErrorDatosRentaFija, MotorRentaFija

bp = Blueprint("renta_fija", __name__, url_prefix="/renta-fija")


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validar_request(data: dict) -> list[str]:
    if not isinstance(data, dict):
        return ["El cuerpo de la petición debe ser un objeto JSON."]
    errores = [f"Falta el campo '{c}'." for c in ("parametros", "instrumentos", "curvas") if c not in data]
    if errores:
        return errores
    for c in ("fechaPortafolio", "fechaDesde"):
        if c not in data["parametros"]:
            errores.append(f"parametros: falta '{c}'.")
    if not data["instrumentos"]:
        errores.append("La lista 'instrumentos' está vacía.")
    for i, instr in enumerate(data["instrumentos"]):
        for c in ("isin", "tipo", "moneda", "curva", "nominal"):
            if c not in instr:
                errores.append(f"Instrumento {i}: falta '{c}'.")
    return errores


@bp.route("/calcular", methods=["POST"])
def calcular():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "El cuerpo de la petición debe ser JSON válido.", "fecha": _ahora()}), 400

    errores = _validar_request(data)
    if errores:
        return jsonify({"error": "Petición inválida", "detalle": errores, "fecha": _ahora()}), 400

    try:
        motor = MotorRentaFija(
            parametros=data["parametros"], instrumentos=data["instrumentos"],
            curvas=data["curvas"], tipos_cambio=data.get("tiposCambio"))
        return jsonify({"fecha": _ahora(), **motor.calcular()}), 200
    except ErrorDatosRentaFija as e:
        return jsonify({"error": str(e), "fecha": _ahora()}), 422
    except Exception as e:
        current_app.logger.exception("Error inesperado calculando renta fija")
        return jsonify({"error": "Error interno al calcular renta fija.", "detalle": str(e), "fecha": _ahora()}), 500


@bp.route("/salud", methods=["GET"])
def salud():
    return jsonify({"estado": "ok", "motor": "renta-fija", "fecha": _ahora()})
