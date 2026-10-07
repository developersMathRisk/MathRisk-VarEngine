"""
motores/stress/rutas.py
=========================
Endpoints HTTP del motor de Stress Testing, como Blueprint independiente (mismo patrón que
motores/var/rutas.py). Registrado en app.py bajo el prefijo "/stress".
"""

from datetime import datetime, timezone

from flask import Blueprint, current_app, jsonify, request

from .motor import ErrorDatosStress, MotorStress

bp = Blueprint("stress", __name__, url_prefix="/stress")


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validar_request(data: dict) -> list[str]:
    errores: list[str] = []
    if not isinstance(data, dict):
        return ["El cuerpo de la petición debe ser un objeto JSON."]

    for campo in ("parametros", "activos", "escenario"):
        if campo not in data:
            errores.append(f"Falta el campo '{campo}'.")
    if errores:
        return errores

    activos = data["activos"]
    if not activos:
        errores.append("La lista 'activos' está vacía.")
    for i, activo in enumerate(activos):
        for campo in ("nombre", "numAcciones", "precioActual", "monedaActivo"):
            if campo not in activo:
                errores.append(f"Activo {i}: falta '{campo}'.")

    escenario = data["escenario"]
    if not isinstance(escenario, dict) or not any(
        k in escenario for k in ("shockPrecioPct", "shockCambiarioPct", "shockPrecioPorActivo")
    ):
        errores.append(
            "El 'escenario' debe indicar al menos uno de: shockPrecioPct, shockCambiarioPct, shockPrecioPorActivo."
        )

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
        motor = MotorStress(parametros=data["parametros"], activos=data["activos"])
        resultado = motor.calcular(escenario=data["escenario"])
        return jsonify({"fecha": _ahora(), **resultado}), 200

    except ErrorDatosStress as e:
        return jsonify({"error": str(e), "fecha": _ahora()}), 422
    except Exception as e:
        current_app.logger.exception("Error inesperado calculando Stress Testing")
        return jsonify({"error": "Error interno al calcular Stress Testing.", "detalle": str(e), "fecha": _ahora()}), 500
