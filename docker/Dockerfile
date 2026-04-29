# syntax=docker/dockerfile:1.6
# LLM Spine Parse Train - PyTorch GPU Docker Image
# Supports: QLoRA training and vLLM serving with LoRA adapters
# Targets: Home GPU (RTX 4060/3090), NRP A100s, Nautilus JupyterHub
#
# Base image: Jupyter GPU-enabled base with proper JupyterHub entrypoint
FROM quay.io/jupyter/pytorch-notebook:latest

# Build args for flexibility
ARG JUPYTERHUB_VERSION=4.0.2

# Switch to root for installations
USER root

# Install additional system dependencies for vision processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Bundle notebook templates and startup bootstrap helper.
COPY notebooks/*.ipynb /opt/llm-spine/notebooks/
COPY docker/scripts/bootstrap-notebooks.sh /usr/local/bin/start-notebook.d/10-bootstrap-notebooks.sh
RUN chmod +x /usr/local/bin/start-notebook.d/10-bootstrap-notebooks.sh && \
    chown -R ${NB_UID}:${NB_GID} /opt/llm-spine

# Switch back to jovyan user for package installations
USER ${NB_UID}

# Ensure JupyterHub version matches Nautilus
RUN pip install --no-cache-dir jupyterhub==${JUPYTERHUB_VERSION}

# Install all project dependencies (training + adapter serving + utils)
COPY docker/requirements.txt /tmp/
RUN --mount=type=cache,target=/home/${NB_USER}/.cache/pip \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Create data directory for model storage
RUN mkdir -p /home/${NB_USER}/data && \
    mkdir -p /home/${NB_USER}/llm_train

# Health check: verify PyTorch GPU and Jupyter
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import torch; import notebook; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" || exit 1

# Default working directory (set to home so Jupyter serves from /home/jovyan/)
WORKDIR /home/${NB_USER}

# The base image already has the proper CMD for JupyterHub compatibility
# For local use, override with docker-compose or `docker run` command
# Default CMD from base image: start-notebook.sh → launches jupyterhub-singleuser or jupyter lab
