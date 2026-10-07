"""
motores/var/rutas.py
=====================
Endpoints HTTP del motor de VaR, como un Blueprint de Flask independiente.
`app.py` solo lo registra bajo el prefijo "/var"; no conoce los detalles de
cómo se valida ni se calcula. Así, agregar un motor nuevo (Stress Testing,
por ejemplo) es crear su propia carpeta en motores/ con este mismo patrón
(motor.py + rutas.py) y registrar su Blueprint en app.py — sin tocar este
archivo ni el de ningún otro motor.
"""

from datetime import datetime, timezone

from flask import Blueprint, current_app, jsonify, request

from .motor import ErrorDatosVaR, METODOLOGIAS_VALIDAS, MotorVaR

bp = Blueprint("var", __name__, url_prefix="/var")


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
    metodologias = params.get("metodologias") or []
    niveles = params.get("nivelesConfianza") or []
    if not metodologias:
        errores.append("parametros.metodologias no puede estar vacío.")
    else:
        invalidas = [m for m in metodologias if m not in METODOLOGIAS_VALIDAS]
        if invalidas:
            errores.append(f"Metodologías no soportadas: {invalidas}. Use: {METODOLOGIAS_VALIDAS}.")
    if not niveles:
        errores.append("parametros.nivelesConfianza no puede estar vacío.")

    activos = data["activos"]
    if not activos:
        errores.append("La lista 'activos' está vacía.")
    for i, activo in enumerate(activos):
        for campo in ("nombre", "numAcciones", "precioActual", "monedaActivo"):
            if campo not in activo:
                errores.append(f"Activo {i}: falta '{campo}'.")

    escenarios = data["escenarios"]
    if len(escenarios) < 31:
        errores.append(f"Se requieren al menos 31 escenarios (se recibieron {len(escenarios)}).")

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
        motor = MotorVaR(
            parametros=data["parametros"],
            activos=data["activos"],
            escenarios=data["escenarios"],
        )
        resultado = motor.calcular(
            metodologias=data["parametros"]["metodologias"],
            niveles_confianza=data["parametros"]["nivelesConfianza"],
            num_simulaciones=int(data["parametros"].get("numSimulaciones", 10000)),
        )
        return jsonify({"fecha": _ahora(), **resultado}), 200

    except ErrorDatosVaR as e:
        return jsonify({"error": str(e), "fecha": _ahora()}), 422
    except Exception as e:  # error inesperado: no se expone el traceback al cliente
        current_app.logger.exception("Error inesperado calculando VaR")
        return jsonify({"error": "Error interno al calcular VaR.", "detalle": str(e), "fecha": _ahora()}), 500
