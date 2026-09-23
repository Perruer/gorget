# Gorget API on CPU. Models run on ONNX Runtime, so the image carries no PyTorch.
# Build from the repository root: docker build -f gorget_api/Dockerfile .
FROM python:3.13-slim

LABEL org.opencontainers.image.source=https://github.com/Perruer/gorget
LABEL org.opencontainers.image.description="Gorget API"
LABEL org.opencontainers.image.licenses=MIT

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR $HOME/app

# The library comes from this repository, so the image matches the commit it is built from.
COPY --chown=user:user pyproject.toml README.md LICENSE ./gorget-src/
COPY --chown=user:user gorget ./gorget-src/gorget
COPY --chown=user:user llm_guard ./gorget-src/llm_guard
COPY --chown=user:user gorget_api/pyproject.toml gorget_api/README.md ./
COPY --chown=user:user gorget_api/app ./app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ./gorget-src && \
    pip install --no-cache-dir . && \
    rm -rf ./gorget-src

COPY --chown=user:user gorget_api/config/scanners.yml ./config/scanners.yml
COPY --chown=user:user gorget_api/entrypoint.sh ./entrypoint.sh

RUN chmod +x ./entrypoint.sh

ENV PYTHONPATH=$HOME/app

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
