# SGLang Spark image for DGX Spark (ARM64/GB10)
# Includes: CUDA 13.0, PyTorch 2.9.0+cu130, FlashInfer 0.4.0, triton 3.5.0
FROM lmsysorg/sglang:spark

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.claude/bin:/root/.local/bin:/usr/local/bin:${PATH}"

# Install uv for package management (install to /usr/local/bin directly)
RUN curl -LsSf https://astral.sh/uv/install.sh | env CARGO_HOME=/usr/local UV_INSTALL_DIR=/usr/local/bin sh

# Install Node.js for Claude Code CLI (native installer has ARM64 bugs)
RUN apt-get update && apt-get install -y nodejs npm && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code

# Ensure PATH is set for interactive shells
RUN echo 'export PATH="/root/.claude/bin:/root/.local/bin:/usr/local/bin:$PATH"' >> /root/.bashrc
