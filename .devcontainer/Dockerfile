# SGLang Spark image for DGX Spark (ARM64/GB10)
# Includes: CUDA 13.0, PyTorch 2.9.0+cu130, FlashInfer 0.5.0, Triton 3.5.0
# Pinned to specific digest for reproducibility
# SGLang version: 0.5.4.post2 (commit 3efd80098e90996458e74677c483ed7ce243b653)
# See: https://github.com/sgl-project/sglang/issues/11658
FROM lmsysorg/sglang:spark@sha256:1d444ecc8bd6b11499a0974b112ef6490c4c89919724a36c0d32e57314de956c

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
