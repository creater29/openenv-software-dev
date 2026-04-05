FROM python:3.11-slim
LABEL maintainer="openenv-team"
LABEL description="OpenEnv Software Development RL Environment"

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl gcc build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache)
COPY pyproject.toml .
RUN pip install --upgrade pip \
    && pip install ".[all]" \
    && pip install fastapi uvicorn[standard]

# Copy project source
COPY . .

# HF Spaces runs as root by default — no need for a separate user
# Expose port 7860 (required by Hugging Face Spaces)
EXPOSE 7860

# Healthcheck: verify the environment + server imports cleanly
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "from openenv_software_dev.env import SoftwareDevEnv; e=SoftwareDevEnv(); e.reset(); print('ok')"

# Launch FastAPI on port 7860 (HF Spaces requirement)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
