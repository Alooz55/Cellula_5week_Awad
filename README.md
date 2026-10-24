# Voice-Based Data Analysis System (Cellula NLP Internship – Project 2)

FastAPI backend (replaces the old Streamlit app) that lets you analyse a dataset **by speaking, in Arabic or English**:

```
Voice → Faster-Whisper → Text → LLM → SQL → validate → SQLite3 → Result + Explanation
```

## Run
```bash
pip install -r requirements.txt
cp .env.example .env          # paste your FREE OpenRouter key (https://openrouter.ai/keys), or set LLM_PROVIDER=ollama
python test_key.py            # optional: check the key
uvicorn main:app --reload     # API docs: http://127.0.0.1:8000/docs   |   demo UI: http://127.0.0.1:8000/
pytest -q                     # offline tests (mock LLM + fake STT, no internet / key needed)
```
The first voice request downloads the Whisper model once (`small` ≈ 480 MB). For better Arabic set `WHISPER_MODEL=medium`.
`ffmpeg` is **not** required (faster-whisper decodes audio with PyAV); wav / mp3 / m4a / webm / ogg all work.

## Endpoints
| Method & path | Purpose |
|---|---|
| `POST /datasets/upload` | upload CSV / TSV / Excel / JSON → SQLite table `data`; returns `dataset_id`, schema, sample rows |
| `GET /datasets/{id}/schema` | columns, types, value hints, sample rows |
| `POST /transcribe` | audio file → text + detected language (Faster-Whisper) |
| `POST /generate-sql` | `{dataset_id, question}` → validated SQL (LLM, with self-correction) |
| `POST /execute-sql` | `{dataset_id, sql}` → validate + run on SQLite3 → columns / rows |
| `POST /text-query` | question text → SQL → result + explanation |
| `POST /voice-query` | **full pipeline**: audio + `dataset_id` → transcript, SQL, rows, explanation |
| `DELETE /datasets/{id}/history` | clear follow-up memory |
| `GET /health` | service check |

```bash
curl -F "file=@data/sales_sample.csv" http://127.0.0.1:8000/datasets/upload          # -> dataset_id
curl -F "audio=@question.wav" -F "dataset_id=<ID>" http://127.0.0.1:8000/voice-query
curl -H "Content-Type: application/json" -d '{"dataset_id":"<ID>","question":"اعرض إجمالي المبلغ لكل فئة"}' \
     http://127.0.0.1:8000/text-query
```
Optional form field `language=ar|en` on `/transcribe` and `/voice-query` forces the language (default: auto-detect).

## How it works (key design points)
- **Arabic + English**: Whisper auto-detects the language; the LLM prompt is bilingual (few-shot in both), receives the
  schema with *all distinct values* of low-cardinality text columns (so Arabic questions map to the right filters),
  Arabic-Indic digits are normalised, Arabic column names are supported, and the explanation is written in the user's language.
- **SQL safety** (`core/sql_runner.py`): single `SELECT`/`WITH` only · forbidden-keyword check (string literals ignored) ·
  DB opened **read-only** (`mode=ro`) · SQLite *authorizer* denies everything except reads · row cap · timeout ·
  double-quoted unknown identifiers raise errors instead of silently becoming strings.
- **Self-correction**: if the SQL is rejected or fails (e.g. hallucinated column), the error is sent back to the LLM (max `MAX_SQL_RETRIES`).
- **Memory**: the last few questions + SQL of each dataset are sent to the LLM so follow-ups ("and for Amman only?") work.
- **Errors**: unsupported/empty/corrupt file, no speech detected, unknown `dataset_id`, bad SQL, LLM down → clear message + proper HTTP code.

## Layout
| file | role |
|---|---|
| `main.py` | FastAPI app, endpoints, error handlers, serves the UI |
| `core/config.py` | env settings (LLM, Whisper, limits) |
| `core/llm.py` | OpenRouter / Ollama client (same free setup as before) |
| `core/stt.py` | Faster-Whisper (model loaded once) |
| `core/datasets.py` | upload → pandas → SQLite, schema description for the prompt |
| `core/sql_generator.py` | prompt, JSON parsing, self-correction prompt |
| `core/sql_runner.py` | validation + safe execution |
| `core/explainer.py` | explanation of the result |
| `core/pipeline.py` | orchestration + conversation memory |
| `static/index.html` | demo page: upload, record with the microphone, results table |
| `data/` | sample datasets (English + Arabic columns) |
| `tests/test_pipeline.py` | 35 offline tests |

## Limits / notes
- One table per dataset (`data`); multi-table joins would need multiple uploads merged into one database.
- Free OpenRouter models are rate-limited (20 req/min, 50/day): each question costs 2 requests (SQL + explanation; +1 per retry).
  Use `"explain": false` or Ollama for unlimited local use.
- Whisper accuracy on strong dialects / noisy audio is lower; use `medium` and speak clearly.
