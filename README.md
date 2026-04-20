# Dementia Companion

This repository contains the public GitHub release for the Dementia Companion project, a multilingual retrieval-augmented generation (RAG) system built to support dementia care information for two related but distinct studies.

The project combines curated web and educational sources, local language-aware retrieval, and study-specific response generation workflows. One study was designed for patients and caregivers, using a local RAG pipeline and frozen index artifacts. The other was designed for experts, using a Perplexity-based workflow for source-backed responses and expert rating.

This repository is split into two folders so each study can be published, cited, and reproduced independently without mixing their logic or inputs.

## Folder Layout

- `patients_caregivers/`: The patient/caregiver web app and RAG pipeline. This folder contains the local answer-generation flow, the index-building script, and the scripts used to persist the shared document index.
- `experts/`: The expert-study web app and Perplexity-based answer flow. This folder contains the study interface used for expert review and source rating.

Each folder includes the core app files needed to run the study interface:

- `main.py`, `database.py`, `ai4hope.yaml`, `.env.example`, `templates/`, and `static/`

Files specific to `patients_caregivers/`:

- `rag.py`
- `build_index_pipeline.sh`, `index_save_to_disk.py`

Files specific to `experts/`:

- `perplexity_api.py`

## What This Release Contains

- The code for both study-facing web applications.
- The documents and scripts needed to reproduce the patient/caregiver index build.
- The expert workflow code for Perplexity-based answering and source collection.
- The configuration files, templates, and static assets required to run the apps.

## Why Two Folders

The expert and patient/caregiver studies are related but not identical:

- They use similar infrastructure.
- They have different interface/flow details.
- They differ in response-generation behavior and evaluation protocol.

Keeping separate folders preserves protocol fidelity for publication and review, and makes it easier to explain exactly which code produced which study results.

## Common Dependencies and Attribution

- Embedding model: [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)
- Crawl preprocessing: Open Web Search EU pipeline image `opencode.it4i.eu:5050/openwebsearcheu-public/preprocessing-pipeline`
- LlamaIndex stack for retrieval and indexing
- Local or hosted model endpoints configured through environment variables

## Publication Split

- GitHub: code, configs, and scripts in this repository.
- Zenodo: frozen `index/` artifacts and other heavy reproducibility assets.

## Using the Repository

If you want to run the patient/caregiver study app, start from `patients_caregivers/`. If you want to run the expert study app, start from `experts/`. The two folders are intentionally similar in structure, but they should be treated as separate study implementations.

## Funding

Funded by the European Union (AI4HOPE, 101136769). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the Health and Digital Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

This work was funded by UK Research and Innovation (UKRI) under the UK government's Horizon Europe funding guarantee [Grant No. 101136769].

## Contact

- izidor.mlakar@um.si
- rigon.sallauka@um.si

