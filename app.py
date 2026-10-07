"""
app.py — host de motores de cálculo
=====================================
Un solo proceso Flask que expone uno o más motores de riesgo, cada uno como
su propio Blueprint bajo su propio prefijo de URL. Hoy solo está registrado
el motor de VaR (motores/var/), pero está preparado para que los demás
motores (Stress Testing, y los que sigan) vivan aquí también sin que el
backend Java tenga que apuntar a una URL distinta por motor.

Para agregar un motor nuevo:
  1. Crear motores/<nombre>/motor.py  (el cálculo puro, sin Flask)
  2. Crear motores/<nombre>/rutas.py  (un Blueprint con url_prefix="/<nombre>")
  3. Importarlo y registrarlo abajo, junto a los demás.
Cada motor queda aislado: un error de validación o de cálculo en uno no
afecta a los otros, y cada uno puede evolucionar su propio contrato JSON.

Ejecutar en desarrollo:
  python app.py                 (http://localhost:5001)

Ejecutar en producción (ejemplo):
  gunicorn -w 2 -b 0.0.0.0:5001 app:app
"""

import hmac
import os
from datetime import datetime, timezone

from flask import Flask, jsonify, request
from flask_cors import CORS

from motores.var.rutas import bp as var_bp
from motores.stress.rutas import bp as stress_bp
from motores.backtesting.rutas import bp as backtesting_bp
from motores.renta_fija.rutas import bp as renta_fija_bp

# Motores registrados: nombre visible -> Blueprint. Se usa también para el /salud general.
MOTORES = [
    ("var", var_bp),
    ("stress", stress_bp),
    ("backtesting", backtesting_bp),
    ("renta-fija", renta_fija_bp),
]

app = Flask(__name__)
CORS(app)  # el backend Java llama a este servicio desde otro origen/puerto

# Clave compartida con el backend. Si MOTOR_API_KEY esta definida, todo pedido (salvo /salud) debe traer el
# encabezado X-Api-Key igual; en un servicio publico (Vercel/Render) la URL es accesible desde internet.
_API_KEY = os.environ.get("MOTOR_API_KEY", "")


@app.before_request
def _exigir_clave():
    if not _API_KEY or request.method == "OPTIONS" or request.path in ("/salud", "/renta-fija/salud"):
        return None
    if not hmac.compare_digest(request.headers.get("X-Api-Key", ""), _API_KEY):
        return jsonify({"error": "No autorizado."}), 401
    return None

for _nombre, _bp in MOTORES:
    app.register_blueprint(_bp)


@app.route("/salud", methods=["GET"])
def salud():
    """Health check general del host. Cada motor puede además tener el suyo (ver motores/<n>/rutas.py)."""
    return jsonify({
        "estado": "ok",
        "servicio": "MathRisk - Motores de Riesgo",
        "version": "1.1.0",
        "motoresDisponibles": [n for n, _ in MOTORES],
        "fecha": datetime.now(timezone.utc).isoformat(),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
