# syntax=docker/dockerfile:1
FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY transcribe.py app.py db.py ./

EXPOSE 8080

# Один worker uvicorn: фоновый пул воркеров живёт в процессе, состояние — в SQLite.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
