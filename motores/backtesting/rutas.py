"""
motores/backtesting/rutas.py
==============================
Endpoints HTTP del motor de Backtesting, como Blueprint independiente (mismo patrón que
motores/var/rutas.py). Registrado en app.py bajo el prefijo "/backtesting".
"""

from datetime import datetime, timezone

from flask import Blueprint, current_app, jsonify, request

from .motor import ErrorDatosBacktesting, MotorBacktesting

bp = Blueprint("backtesting", __name__, url_prefix="/backtesting")


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validar_request(data: dict) -> list[str]:
    errores: list[str] = []
    if not isinstance(data, dict):
        return ["El cuerpo de la petición debe ser un objeto JSON."]

    for campo in ("parametros", "activos", "escenarios"):
        if campo not in data:
            errores.append(f"Falta el campo '{campo}'.")
    if errores:
        return errores

    params = data["parametros"]
    if "nivelConfianza" not in params:
        errores.append("parametros.nivelConfianza es obligatorio.")
    if "ventana" not in params:
        errores.append("parametros.ventana es obligatorio.")

    activos = data["activos"]
    if not activos:
        errores.append("La lista 'activos' está vacía.")
    for i, activo in enumerate(activos):
        for campo in ("nombre", "numAcciones", "precioActual", "monedaActivo"):
            if campo not in activo:
                errores.append(f"Activo {i}: falta '{campo}'.")

    escenarios = data["escenarios"]
    try:
        minimo = int(params.get("ventana", 31)) + 10
    except (TypeError, ValueError):
        minimo = 41
    if len(escenarios) < minimo:
        errores.append(f"Se requieren al menos {minimo} escenarios para esta ventana (se recibieron {len(escenarios)}).")

    nombres_activos = {a["nombre"] for a in activos if "nombre" in a}
    for esc in escenarios:
        if "numero" not in esc or "precios" not in esc:
            errores.append(f"Escenario inválido (falta 'numero' o 'precios'): {esc}")
            continue
        faltantes = nombres_activos - set(esc["precios"].keys())
        if faltantes:
            errores.append(f"Escenario #{esc['numero']}: faltan precios para {sorted(faltantes)}.")

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
        motor = MotorBacktesting(
            parametros=data["parametros"], activos=data["activos"], escenarios=data["escenarios"],
        )
        resultado = motor.calcular()
        return jsonify({"fecha": _ahora(), **resultado}), 200

    except ErrorDatosBacktesting as e:
        return jsonify({"error": str(e), "fecha": _ahora()}), 422
    except Exception as e:
        current_app.logger.exception("Error inesperado calculando Backtesting")
        return jsonify({"error": "Error interno al calcular Backtesting.", "detalle": str(e), "fecha": _ahora()}), 500
