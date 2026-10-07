# Motores de Riesgo (Python)

Un solo microservicio Flask que hospeda los motores de cálculo de riesgo.
Hoy solo tiene el motor de **VaR** (acciones y fondos mutuos, tres
metodologías: **Histórico**, **Monte Carlo** y **Paramétrico**, sobre la
misma ventana de escenarios de precio para que los resultados sean
comparables entre sí), pero está organizado para sumar más motores (por
ejemplo, Stress Testing) sin reescribir nada de lo que ya existe.

El motor de VaR está adaptado del validado en
`Motor Var Acciones/Motor API/var_api_flask.py` (que reproduce la plantilla
Excel "VaR Acciones o Fondos Mutuos").

## Arquitectura

```
Angular  →  Backend Java (Spring)  →  JSON (portafolio + precios + TC)  →  este host Flask  →  motor correspondiente
```

Este servicio **no se conecta a ninguna base de datos**. El backend Java ya
tiene acceso a los precios históricos y tipos de cambio en PostgreSQL
(tablas de factores de riesgo); arma el JSON de entrada y llama a este
servicio por HTTP. Así cada motor queda desacoplado y se puede probar con
cualquier fuente de datos, incluida la misma plantilla Excel.

### Un proceso, varios motores

```
MathRisk-VarEngine/
  app.py                  # crea la app Flask y registra un Blueprint por motor
  motores/
    var/
      motor.py            # el cálculo puro (MotorVaR): sin Flask, sin HTTP
      rutas.py            # Blueprint: valida el JSON y expone POST /var/calcular
    stress/                # (futuro) mismo patrón: motor.py + rutas.py
      ...
```

`app.py` no sabe nada de cómo calcula cada motor: solo importa su Blueprint
y lo registra bajo su propio prefijo de URL (`/var`, y mañana `/stress`).
El backend Java sigue apuntando a **una sola URL base** (`varengine.url` en
`application.properties`) sea cual sea el motor que llame, cambiando solo
el path (`/var/calcular`, `/stress/calcular`, …). Agregar un motor nuevo es:

1. Crear `motores/<nombre>/motor.py` con la lógica de cálculo (clases y
   funciones normales de Python, sin ninguna dependencia de Flask —
   así se puede probar e importar igual que `motores/var/motor.py`).
2. Crear `motores/<nombre>/rutas.py` con un `Blueprint(..., url_prefix="/<nombre>")`
   que valide el JSON de entrada y llame a `motor.py` (ver `motores/var/rutas.py`
   como plantilla).
3. Registrarlo en la lista `MOTORES` de `app.py` (una línea).

Un motor con errores de validación o de cálculo no puede tumbar a otro: cada
uno vive en su propio Blueprint con su propio manejo de errores.

## Puesta en marcha

```bash
cd MathRisk-VarEngine
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python app.py                   # http://localhost:5001
```

En producción, detrás de un proceso administrado:

```bash
gunicorn -w 2 -b 0.0.0.0:5001 app:app
```

## Endpoints

### `GET /salud`
Health check del host (no de un motor en particular). Devuelve
`motoresDisponibles` con la lista de motores registrados. Úsalo para el
probe de arranque del backend Java.

### `POST /var/calcular`

Cuerpo de la petición:

```json
{
  "parametros": {
    "monedaReporte": "PEN",
    "tcActual": { "USD_PEN": 3.75 },
    "horizonteDias": 1,
    "metodologias": ["historico", "montecarlo", "parametrico"],
    "nivelesConfianza": [0.95, 0.975, 0.99],
    "numSimulaciones": 10000
  },
  "activos": [
    { "nombre": "AMZN", "numAcciones": 848, "precioActual": 25.54, "monedaActivo": "USD" }
  ],
  "escenarios": [
    {
      "numero": 252,
      "fecha": "2024-01-02",
      "precios": { "AMZN": 24.10 },
      "tcHistorico": { "USD_PEN": 3.71 }
    }
  ]
}
```

Notas sobre `escenarios`:
- Se necesitan al menos 31 (recomendado 252, un año de pregones).
- `numero` es el orden cronológico: el escenario más antiguo tiene el
  número más alto y el más reciente el número 1 (así lo entrega
  `db_loader.py` de la prueba de concepto original). El motor los reordena
  internamente, así que el orden en el arreglo no importa.
- `tcHistorico` es opcional; si falta para una fecha, se usa
  `parametros.tcActual` (tipo de cambio spot) para ese escenario.

Respuesta (200):

```json
{
  "fecha": "2026-09-22T05:24:33Z",
  "mtmTotal": 56897.07,
  "mtmPorActivo": { "AMZN": 21567.89, "...": 0 },
  "numEscenarios": 259,
  "horizonteDias": 1,
  "monedaReporte": "PEN",
  "tiempoCalculoMs": 4.2,
  "resultados": [
    {
      "metodologia": "historico",
      "nivelConfianza": 0.95,
      "varDiversificado": -1111.19,
      "varNoDiversificado": -1832.33,
      "beneficioDiversificacion": -721.15,
      "varIndividual": { "AMZN": -663.1, "...": 0 },
      "varDesagregado": { "AMZN": -420.3, "...": 0 }
    }
  ],
  "distribucionHistorica": [-320.1, 145.6, -80.2]
}
```

`distribucionHistorica` solo aparece si `metodologias` incluye `"historico"`: es el P&L
total del portafolio en cada uno de los `numEscenarios` escenarios reales (mismo orden
que los escenarios de entrada, del más antiguo al más reciente). El backend Java la usa
para guardar y graficar el histograma de pérdidas/ganancias. No se calcula para Monte
Carlo (miles de simulaciones sintéticas sin fecha real detrás).

Todos los montos de VaR se expresan como **pérdida** (número negativo) en
`monedaReporte`. `varDesagregado` es la contribución de cada activo al VaR
diversificado del portafolio:
- En **histórico** y **Monte Carlo**, es el P&L de cada activo en el
  escenario/simulación donde ocurre la pérdida del percentil elegido.
- En **paramétrico**, es la asignación de Euler (`z · wᵢ · (Σw)ᵢ / σ`),
  que por construcción suma exactamente el VaR diversificado.

Errores:
- `400` — la petición no es JSON válido o le faltan campos obligatorios
  (el cuerpo incluye `detalle` con la lista de problemas).
- `422` — los datos son válidos pero insuficientes para calcular
  (por ejemplo, un escenario sin precio para algún activo).
- `500` — error inesperado del motor.

## Metodologías

Las tres parten de la misma matriz de variaciones combinadas
(precio × tipo de cambio) construida a partir de los escenarios:

| Metodología  | Cómo estima la cola de pérdida |
|--------------|---------------------------------|
| Histórico    | Percentil empírico de la distribución real de P&L (no asume ninguna forma de distribución). |
| Monte Carlo  | `numSimulaciones` simulaciones de una normal multivariada con la media y covarianza históricas (conserva las correlaciones entre activos), semilla fija (reproducible). |
| Paramétrico  | Delta-normal: `VaR = z_(1-confianza) · σ_portafolio`, con `σ_portafolio² = wᵀΣw` (w = exposición en dinero por activo, Σ = covarianza de las variaciones). |

`k_esimo` (el percentil de pérdida) se calcula igual que en la plantilla
Excel: `K = 100 · (1 − nivel_confianza)`.

## Pruebas rápidas

No hay suite de pruebas automatizada todavía. Para una verificación manual:

```bash
python -c "
from motores.var.motor import MotorVaR
# ver el bloque de ejemplo en la sección 'Endpoints' de este README
"
```

o levantar `app.py` y enviar el ejemplo de `POST /var/calcular` con `curl`
o Postman.

## Carpetas de referencia (no desplegadas)

El resto de `Motor Var Acciones/` (Motor optimizado, Motor Parametrico,
plantillas Excel, scripts de BD) se queda donde está; sirvió para diseñar
y validar este motor pero no forma parte del despliegue.

## Motor de renta fija (`/renta-fija`)

Valoriza CD y bonos y calcula VaR/CVaR por simulación histórica de curvas y tipo de cambio (portado de
`PrometheusModelos / VaR - 190926`). Sin estado: todo llega por JSON (contrato en `motores/renta_fija/rutas.py`).

- `POST /renta-fija/calcular` · `GET /renta-fija/salud`
- Pruebas: `python -m unittest tests.test_renta_fija`
- Regresión contra el original (necesita sus archivos): `python tests/regresion_renta_fija.py "<ruta a 'VaR - 190926'>"`
  — MTM y P&L por escenario coinciden con el original (diferencia máxima ~1e-8) en CD02FEB21, CD05NOV20,
  SB12AGO24 y US715638AS19.
