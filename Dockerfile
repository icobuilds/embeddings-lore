# ── Stage 1: dependency installer ────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install deps first (layer-cached until lockfile changes)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Install the project package itself (hatchling requires README.md at build time)
COPY README.md ./
COPY app/ app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.13-slim-bookworm AS runtime

WORKDIR /app

# Copy the fully-built virtualenv from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Application code
COPY app/ app/

# Config files (no .env — credentials come in as env vars at runtime)
COPY config/ config/

COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

ENV PATH="/app/.venv/bin:$PATH" \
    # Default to the production config; override with EMBLORE_CONFIG_PATH at runtime
    EMBLORE_CONFIG_PATH=config/config.yaml \
    # Keep model downloads in a predictable, volume-mountable location
    HF_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
