# amorphouspy-api service

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
COPY LICENSE .

COPY amorphouspy amorphouspy/
RUN micromamba run pip install /app/amorphouspy

COPY amorphouspy_api/ amorphouspy_api/
RUN micromamba run pip install /app/amorphouspy_api

CMD ["uvicorn","amorphouspy_api.app:app","--host", "0.0.0.0","--port", "8000","--log-level","info","--backlog","1","--timeout-keep-alive","60"]