# Dementia Companion Publication Bundle

This repository is organized by study protocol to keep reproducibility clear.

## Folder Layout

- `patients_caregivers/`: Patient/caregiver study implementation (local RAG with frozen index).
- `experts/`: Expert-study implementation (Perplexity-assisted expert workflow plus study-specific logic).

Each folder contains its own:

- `main.py`, `database.py`, `ai4hope.yaml`, `.env.example`, `templates/`, and `static/`

Files specific to `patients_caregivers/`:

- `rag.py`
- `build_index_pipeline.sh`, `index_save_to_disk.py`

Files specific to `experts/`:

- `perplexity_api.py`

## Why Two Folders

The expert and patient/caregiver studies are related but not identical:

- They use similar infrastructure.
- They have different interface/flow details.
- They differ in response-generation behavior and evaluation protocol.

Keeping separate folders preserves protocol fidelity for publication and review.

## Common Dependencies and Attribution

- Embedding model: [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)
- Crawl preprocessing: Open Web Search EU pipeline image `opencode.it4i.eu:5050/openwebsearcheu-public/preprocessing-pipeline`
- LlamaIndex stack for retrieval and indexing

## Publication Split

- GitHub: code, configs, and scripts in this repository.
- Zenodo: frozen `index/` artifacts and other heavy reproducibility assets.

