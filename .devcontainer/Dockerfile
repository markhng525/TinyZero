# NGC PyTorch with ARM64 support for DGX Spark
# Note: hiyouga/verl is x86_64 only, so we use NGC and install deps
FROM nvcr.io/nvidia/pytorch:25.09-py3

# Install uv for package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -sf /root/.local/bin/uv /usr/local/bin/uv

# Build tools for flash-attn compilation (if needed)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:${PATH}"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Unset NGC constraint file for custom package versions
ENV PIP_CONSTRAINT=""
