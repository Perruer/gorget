# API Deployment

## From source

1. Copy the code from [gorget_api](https://github.com/Perruer/gorget/tree/main/gorget_api)

2. Install dependencies (preferably in a virtual environment)
```bash
python -m pip install .
python -m pip install ".[gpu]" # PyTorch, if you have a GPU
```

3. Alternatively, you can use Makefile:
```bash
make install
```

### Using uvicorn

Run the API locally:

```bash
make run
```

Or using CLI:

```bash
gorget_api ./config/scanners.yml
```

### Using gunicorn

In case you want to use `gunicorn` to run the API, you can use the following command:

```bash
gunicorn --workers 1 --preload --worker-class uvicorn.workers.UvicornWorker 'app.app:create_app(config_file="./config/scanners.yml")'
```

It will preload models in the shared memory among workers, which can be useful for performance.

## From Docker

Pull the image from GitHub Container Registry:

```bash
docker pull ghcr.io/perruer/gorget-api:latest       # CPU, ONNX Runtime, no PyTorch
docker pull ghcr.io/perruer/gorget-api:latest-cuda  # NVIDIA GPU, PyTorch
```

Or build it from the repository root:

```bash
docker build -f gorget_api/Dockerfile -t gorget-api .
docker build -f gorget_api/Dockerfile-cuda -t gorget-api:cuda .
```

Now, you can run the Docker container:

```bash
docker run -d -p 8000:8000 -e LOG_LEVEL='DEBUG' -e AUTH_TOKEN='my-token' ghcr.io/perruer/gorget-api:latest
```

This will start the API on port 8000. You can now access the API at `http://localhost:8000/swagger.json`.

If you want to use a custom configuration, you can mount a volume to `/home/user/app/config`:

```bash
docker run -d -p 8000:8000 -e APP_WORKERS=1 -e AUTH_TOKEN='my-token' -e LOG_LEVEL='DEBUG' -v ./entrypoint.sh:/home/user/app/entrypoint.sh -v ./config/scanners.yml:/home/user/app/config/scanners.yml ghcr.io/perruer/gorget-api:latest
```

!!! tip

    Memory grows with the number of model-based scanners. The CPU image carries no PyTorch; scanners
    that share a model (for example `Language` and `LanguageSame`) share one ONNX session. To keep
    models out of the network path at startup, mount a filled Hugging Face cache and set
    `HF_HUB_OFFLINE=1`.

## Troubleshooting

### Out-of-memory error

If you get an out-of-memory error, you can change `config.yml` file to use less scanners.
Alternatively, you can enable `low_cpu_mem_usage` in scanners that rely on HuggingFace models.

### Failed HTTP probe

If you get a failed HTTP probe, it might be because the API is still starting. You can increase the `initialDelaySeconds` in the Kubernetes deployment.

Alternatively, you can configure `lazy_load` in the YAML config file to load models only on the first request.
