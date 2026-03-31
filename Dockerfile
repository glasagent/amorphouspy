# amorphouspy-api service
FROM ghcr.io/prefix-dev/pixi:0.46.0-jammy-cuda-12.6.3 AS build

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    vim \
    procps \
    && apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pixi.toml pixi.lock ./
RUN pixi install --locked

COPY LICENSE .
COPY pyproject.toml .
COPY amorphouspy amorphouspy/
COPY amorphouspy_api/ amorphouspy_api/
RUN pixi run -- pip install --no-deps --no-build-isolation .

EXPOSE 8000

CMD ["pixi", "run", "serve"]