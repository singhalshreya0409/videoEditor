FROM python:3.11-slim
WORKDIR /app

# Install system dependencies (ffmpeg for stream-copy chunking)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY deploy/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/

WORKDIR /app/backend
ENV PORT=8080
ENV PYTHONPATH=/app/backend
CMD ["python", "server.py"]
