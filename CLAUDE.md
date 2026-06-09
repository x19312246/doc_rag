# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A local-first RAG (Retrieval-Augmented Generation) system that ingests PDF documents, indexes them into a ChromaDB vector store, and answers queries via locally-hosted LLMs (Ollama / LM Studio) or Groq cloud. Designed to run fully offline once model weights are cached.

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Start the Flask server (accessible at http://127.0.0.1:5000)
python app_flask.py
```

There is no test suite. Validation is done by running the app and exercising its API endpoints.

## Architecture

### Request Flow

```
Browser UI → Flask API → Background Thread Worker → ChromaDB / LLM
```

Long-running operations (OCR, VLM reconstruction, RAG query) are dispatched from Flask route handlers into daemon threads. Task state is tracked in the global `TASK_STATUS` dict. The frontend polls `/api/task_status/<task_type>` for status; polling noise is filtered from logs.

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `app_flask.py` | Flask routes, background worker launchers, task cancellation tokens |
| `config/settings.py` | All paths (`CHROMADB_DIR`, `RAW_DATA_DIR`, `MODEL_WEIGHTS_ROOT`, etc.) and model names; detects dev vs PyInstaller env; auto-enables HF offline mode when weights exist |
| `indexer/ocr_loader.py` | PDF → page images (pdf2image), text (pdfplumber + pytesseract fallback), tables (img2table), per-page image crops; VLM re-reconstruction pipeline |
| `indexer/indexer.py` | Upserts chunk dicts into ChromaDB |
| `retriever/retriever.py` | Two-stage retrieval: embedding query (top 25) → CrossEncoder rerank (top 12); special forced full-page range fetching for Chinese page-range queries |
| `model/embeddings.py` | `ChromaEmbeddingFunction` wrapping `intfloat/multilingual-e5-large-instruct` |
| `model/rerank.py` | `BAAI/bge-reranker-base` CrossEncoder via `sentence-transformers` |
| `model/llm.py` | `query_llm()` routing to Groq, LM Studio, or Ollama via OpenAI-compatible API |

### ChromaDB Collection Naming

Each document gets its own collection: `collection_{md5(filename_without_ext)}`. The doc ID is computed in `app_flask.py:api_check_file()` and passed through all subsequent pipeline calls.

### Chunk Types

Each indexed page produces up to four chunk types stored in metadata `"type"` field:
- `"text"` — main body text, split at 1000 chars / 200 overlap
- `"table"` — Markdown-formatted table from img2table
- `"image"` — anchor chunk referencing an extracted embedded image
- `"vlm_text"` — VLM-reconstructed text added after a second pass (optional)
- `"global_summary"` — one summary chunk per document covering all pages

### Task Cancellation

Each task type (`ocr`, `vlm`, `query`) has an `ACTIVE_CANCELLATIONS[task_type]` token (a `TaskCancellation` object). Workers check `cancel_token.is_running` at checkpoints. `POST /api/cancel/<task_type>` sets it to `False`.

## Configuration

### Model Weights (Offline Mode)

Place cached HuggingFace weights under:
- `model_weights/embed/` — `intfloat/multilingual-e5-large-instruct`
- `model_weights/rerank/` — `BAAI/bge-reranker-base`

When both dirs contain valid weights, `config/settings.py` sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` automatically.

### LLM Providers

Supported providers (passed as `provider` in API requests):
- `"ollama"` — local Ollama at `http://<ip>:<port>/v1` (default port 11434)
- `"lmstudio"` — local LM Studio at `http://<ip>:<port>/v1`
- `"groq"` — Groq cloud at `https://api.groq.com/openai/v1` (requires `api_key`)

Provider strings are normalized to lowercase before routing in both `app_flask.py` and `model/llm.py`.

### OCR

Tesseract is expected to be installed system-wide, or placed at `Tesseract-OCR/` under the project root. Tessdata can be placed at `Tesseract/`. Default OCR language: `chi_tra+eng`.

### PyInstaller Packaging

`config/settings.py` and `app_flask.py` both detect `sys.frozen` (PyInstaller bundle) and resolve paths from `sys._MEIPASS` instead of `__file__`.
