# Container image for adfusion (full pipeline, CPU-only PyTorch).
#
#   docker build --target test    -t adfusion-test .   # package, tests, lint tools
#   docker build --target runtime -t adfusion .        # the adfusion command only
#   docker run --rm adfusion --version
#
# The image contains code only. Imaging data are mounted at run time
# (docker run -v /path/to/work:/data ...) and never copied into an image.
# For a CUDA build pass --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126

ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------- base
FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg
WORKDIR /app

# ---------------------------------------------------------------- deps: PyTorch
# Installed in its own layer, so code changes do not download PyTorch again.
FROM base AS deps
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install torch --index-url "${TORCH_INDEX}"

# ---------------------------------------------------------------- build: wheel
FROM base AS build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip wheel --no-deps --wheel-dir /dist .

# ---------------------------------------------------------------- test
FROM deps AS test
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install -e ".[train,dev]"
COPY tests ./tests
COPY scripts ./scripts
COPY configs ./configs
CMD ["pytest", "-q"]

# ---------------------------------------------------------------- runtime
FROM deps AS runtime
ARG VERSION=dev
ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="adfusion" \
      org.opencontainers.image.description="sMRI + FDG-PET fusion pipeline for Alzheimer's disease classification" \
      org.opencontainers.image.source="https://github.com/Octozavrus/multimodal-ad-fusion" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}"
COPY --from=build /dist/*.whl /tmp/
RUN pip install "$(ls /tmp/adfusion-*.whl)[train]" \
 && rm /tmp/*.whl \
 && useradd --create-home --uid 1000 adfusion
COPY configs /app/configs
USER adfusion
WORKDIR /work
ENTRYPOINT ["adfusion"]
CMD ["--help"]
