# Motores de riesgo (Flask). Un solo proceso; 1 worker por la memoria de los planes gratuitos.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY motores ./motores
EXPOSE 5001
CMD ["sh", "-c", "gunicorn -w 1 --threads 4 -b 0.0.0.0:${PORT:-5001} app:app"]
