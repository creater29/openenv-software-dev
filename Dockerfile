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
    && pip install ".[all]"

# Copy project source
COPY . .

# Create a non-root user for safety
RUN useradd --create-home --shell /bin/bash agent \
    && chown -R agent:agent /app
USER agent

# Healthcheck: verify the environment imports cleanly
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "from openenv_software_dev.env import SoftwareDevEnv; e=SoftwareDevEnv(); e.reset(); print('ok')"

# Default: run the random baseline agent
CMD ["python", "inference.py"]
