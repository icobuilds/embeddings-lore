# embeddings-lore

A self-hosted, server-deployable FastAPI application that loads multiple open-source embedding models, indexes a candidate database, and exposes similarity search with configurable metadata fields + benchmarks each model on speed, memory, and retrieval quality.

---

## Docker

### Local (with `.env`)

```bash
docker build -t embeddings-lore .

docker run -p 8000:8000 \
  -e EMBLORE_DB__USER=<user> \
  -e EMBLORE_DB__PASSWORD=<password> \
  -e EMBLORE_DB__DATABASE=<database> \
  -e EMBLORE_DB__TABLE=<table> \
  -e EMBLORE_CONFIG_PATH=config/config.yaml \
  -v hf-cache:/app/.cache/huggingface \
  -v faiss-indexes:/app/indexes \
  embeddings-lore
```

### First run: build indexes

On a fresh deployment the `indexes/` volume is empty. Load a model and trigger an index build:

```bash
# Load a model into memory
curl -X POST http://localhost:8000/models/load \
  -H "Content-Type: application/json" \
  -d '{"name": "multilingual-e5-small"}'

# Build its FAISS index (runs in the background)
curl -X POST "http://localhost:8000/index/build?model=multilingual-e5-small&async_mode=true"

# Poll until complete
curl http://localhost:8000/index/status
```

Once built, the index is written to the mounted volume and automatically reloaded on the next container start.

---

## Local development
Credentials and config path should be loaded automatically from `.env`
Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # fill in DB credentials
uvicorn app.main:app --reload
```

---

## API overview

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check, lists configured and loaded models |
| `GET` | `/models` | All configured models with load status and embedding dim |
| `POST` | `/models/load` | Load a model into memory |
| `DELETE` | `/models/{name}` | Unload a model from memory |
| `POST` | `/index/build` | Build (or rebuild) the FAISS index for a model |
| `GET` | `/index/status` | Build metadata for every configured model |
| `POST` | `/similarity` | Top-K similarity search against a model's index |
