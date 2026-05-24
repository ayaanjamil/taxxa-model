"""
Modal web endpoint for the Taxxa FastAPI server.

Deploy:
    modal deploy serve.py

The app mounts the existing Modal volumes so no data needs to be
re-uploaded — vectors, BM25 index, nodes, and graph are read from
the taxxa-embed volume; BGE-M3 model weights come from taxxa-hf-cache.

Secrets (set once in the Modal dashboard as "taxxa-secrets"):
    OPENROUTER_API_KEY
    TAXXA_EMBED_MODEL=bge-m3
    TAXXA_DATA_DIR=/data
    TAXXA_GRAPH_VERSION=v2
    TAXXA_ANSWER_MODEL=anthropic/claude-haiku-4-5
    TAXXA_PLANNER_MODEL=anthropic/claude-haiku-4-5
"""

import modal

APP_NAME = "taxxa-server"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libvoikko1", "voikko-fi")
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("api", "retriever", "answerer", "embedder")
)

app = modal.App(APP_NAME, image=image)

data_vol = modal.Volume.from_name("taxxa-embed")
hf_vol = modal.Volume.from_name("taxxa-hf-cache")


@app.function(
    gpu="A10G",
    volumes={
        "/data": data_vol,
        "/root/.cache/huggingface": hf_vol,
    },
    memory=32768,
    min_containers=1,
    timeout=600,
    secrets=[modal.Secret.from_name("taxxa-secrets")],
)
@modal.asgi_app()
def fastapi_app():
    from api import app as _app
    return _app
