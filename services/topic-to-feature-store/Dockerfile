# --------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- Stage 1: Buildiing the image -------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

# Use the official Python 3.10 slim image
FROM python:3.10-slim AS builder
#FROM python:3.10-alpine AS builder

# Install Python Poetry 1.8.3
#RUN pip install poetry==1.8.3

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set the working directory in the container
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev

# Install Poetry
RUN pip install --no-cache-dir poetry==1.8.3


COPY pyproject.toml poetry.lock /app/

# Install the dependencies
# RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --no-dev --no-root
RUN poetry install --no-dev --no-root && rm -rf $POETRY_CACHE_DIR

# Remove unnecessary build tools to reduce image size
RUN apt-get purge -y --auto-remove gcc  python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . .

# --------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------- Stage 2: Runtime stage ----------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

FROM python:3.10-slim AS runtime

WORKDIR /app
# Set environment variables for runtime stage
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder /app /app

ENTRYPOINT ["python", "-m", "src.main"]