# pyiron-glass-api service

# Note: Using ubuntu-based micromamba images, since default debian-based micromamba images 
# don't work with http proxy settings,
FROM mambaorg/micromamba:1.5.9-jammy AS build

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    vim \
    procps \
    && apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER mambauser

# Install the environment
COPY environment.yml .

RUN micromamba install -y -n base -f environment.yml && \
    micromamba clean --all --yes


EXPOSE 8000

WORKDIR /app
COPY pyiron_glass pyiron_glass/
COPY pyiron_glass_api/ pyiron_glass_api/
COPY LICENSE .
RUN micromamba run pip install /app/pyiron_glass
RUN micromamba run pip install /app/pyiron_glass_api

ENV PYIRONRESOURCEPATH "/app/pyiron_glass_api/scratch"
ENV PYIRONSQLCONNECTIONSTRING "sqlite:////app/pyiron_glass_api/scratch/pyiron.db"

CMD ["uvicorn","pyiron_glass_api.app:app","--host", "0.0.0.0","--port", "8000","--log-level","trace","--backlog","1","--timeout-keep-alive","60"]