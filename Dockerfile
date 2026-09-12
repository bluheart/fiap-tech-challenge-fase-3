FROM python:3.13-slim

WORKDIR /app

# Install system dependencies needed for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    make \
    && rm -rf /var/lib/apt/lists/*


# Install uv for faster package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy application
COPY . .

RUN uv sync --no-dev

# Criar diretório para logs
RUN mkdir -p /app/logs

EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]