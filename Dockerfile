FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/backend

WORKDIR /app

# Install Python backend dependencies (pin scikit-learn==1.9.1 for exact joblib artifact compatibility)
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir scikit-learn==1.9.1 && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend + optional shared ml/ and artifacts/ directories (excluded dirs filtered by .dockerignore)
COPY . /app

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
