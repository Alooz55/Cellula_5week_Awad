"""FastAPI backend: Voice -> Faster-Whisper -> text -> LLM -> SQL -> SQLite3 -> analysis result.

Run:  uvicorn main:app --reload      (docs at http://127.0.0.1:8000/docs, UI at http://127.0.0.1:8000/)
"""
from __future__ import annotations

import functools
import logging
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from core import config, pipeline, stt
from core.datasets import DatasetError, DatasetNotFound, create_dataset, describe
from core.llm import get_llm
from core.sql_runner import SQLExecutionError, SQLValidationError, execute_sql

log = logging.getLogger("voice-sql")
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Voice-Based Data Analysis System",
              description="Speak (Arabic or English) -> Faster-Whisper -> LLM -> SQL -> SQLite3 -> result + explanation",
              version="1.0.0")


# ----------------------------------------------------------------- dependencies
@functools.lru_cache(maxsize=1)
def _default_llm():
    return get_llm()


def llm_dep():
    """Overridable in tests via app.dependency_overrides[llm_dep]."""
    try:
        return _default_llm()
    except (RuntimeError, ValueError) as e:
        raise HTTPException(503, f"LLM is not configured: {e}")


def transcriber_dep():
    """Returns a callable (bytes, filename, language) -> stt.Transcript. Overridable in tests."""
    return stt.transcribe_bytes


# ----------------------------------------------------------------- error handling
@app.exception_handler(DatasetNotFound)
async def _nf(_: Request, e: DatasetNotFound):
    return JSONResponse({"detail": str(e)}, status_code=404)


@app.exception_handler(DatasetError)
async def _de(_: Request, e: DatasetError):
    return JSONResponse({"detail": str(e)}, status_code=400)


@app.exception_handler(SQLValidationError)
async def _sv(_: Request, e: SQLValidationError):
    return JSONResponse({"detail": str(e)}, status_code=400)


@app.exception_handler(SQLExecutionError)
async def _se(_: Request, e: SQLExecutionError):
    return JSONResponse({"detail": f"SQL execution failed: {e}"}, status_code=422)


@app.exception_handler(pipeline.PipelineError)
async def _pe(_: Request, e: pipeline.PipelineError):
    return JSONResponse({"detail": str(e), **e.context}, status_code=e.status)


@app.exception_handler(Exception)
async def _unhandled(_: Request, e: Exception):
    # LLM/network failures end up here (bad key, rate limit, Ollama down...)
    log.exception("Unhandled error")
    return JSONResponse({"detail": f"Upstream or internal error: {type(e).__name__}: {e}"}, status_code=502)


# ----------------------------------------------------------------- schemas
class QuestionIn(BaseModel):
    dataset_id: str
    question: str = Field(min_length=1)
    explain: bool = True


class SQLIn(BaseModel):
    dataset_id: str
    sql: str = Field(min_length=1)


# ----------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    return {"status": "ok", "llm_provider": config.LLM_PROVIDER, "whisper_model": config.WHISPER_MODEL}


@app.post("/datasets/upload", summary="1) Upload a dataset (CSV/TSV/Excel/JSON) -> SQLite")
async def upload_dataset(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise DatasetError("Uploaded file is empty.")
    return create_dataset(file.filename or "data.csv", raw).to_dict()


@app.get("/datasets/{dataset_id}/schema", summary="Columns, types, value hints and sample rows")
def dataset_schema(dataset_id: str):
    return describe(dataset_id).to_dict()


@app.delete("/datasets/{dataset_id}/history", summary="Forget the conversation memory of this dataset")
def clear_history(dataset_id: str):
    describe(dataset_id)
    pipeline.clear_history(dataset_id)
    return {"cleared": True}


def _read_audio(audio: UploadFile, data: bytes) -> None:
    if not data:
        raise HTTPException(400, "Uploaded audio file is empty.")
    if len(data) > config.MAX_AUDIO_MB * 1024 * 1024:
        raise HTTPException(413, f"Audio too large (max {config.MAX_AUDIO_MB} MB).")


def _transcribe(transcriber, audio: UploadFile, data: bytes, language: str | None):
    _read_audio(audio, data)
    lang = language if language in ("ar", "en") else None
    try:
        t = transcriber(data, audio.filename or "audio.wav", lang)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(422, f"Could not decode the audio file (unsupported or corrupt): {e}")
    if not t.text.strip():
        raise HTTPException(422, "No speech detected in the audio. Please speak clearly and try again.")
    return t


@app.post("/transcribe", summary="2) Speech -> text (Faster-Whisper, Arabic/English)")
async def transcribe(audio: UploadFile = File(...), language: str | None = Form(None),
                     transcriber=Depends(transcriber_dep)):
    t = await run_in_threadpool(_transcribe, transcriber, audio, await audio.read(), language)
    return {"text": t.text, "language": t.language, "language_probability": round(t.language_probability, 3),
            "duration_sec": round(t.duration, 2)}


@app.post("/generate-sql", summary="3) Question (Arabic/English) -> validated SQL, not returned as data")
def generate_sql_endpoint(body: QuestionIn, llm=Depends(llm_dep)):
    info = describe(body.dataset_id)
    sql, _, attempts = pipeline.generate_valid_sql(llm, info, body.question, pipeline.get_history(body.dataset_id))
    return {"question": body.question, "sql": sql, "sql_retries": attempts}


@app.post("/execute-sql", summary="4) Validate + execute a SELECT on SQLite3 (read-only)")
def execute_sql_endpoint(body: SQLIn):
    return execute_sql(body.dataset_id, body.sql).to_dict()


@app.post("/text-query", summary="Text question -> SQL -> result + explanation (no audio)")
def text_query(body: QuestionIn, llm=Depends(llm_dep)):
    return pipeline.run_text_query(llm, body.dataset_id, body.question, body.explain)


@app.post("/voice-query", summary="FULL PIPELINE: audio -> text -> SQL -> SQLite -> result + explanation")
async def voice_query(audio: UploadFile = File(...), dataset_id: str = Form(...), language: str | None = Form(None),
                      explain: bool = Form(True), llm=Depends(llm_dep), transcriber=Depends(transcriber_dep)):
    describe(dataset_id)                                   # fail fast (404) before the slow STT step
    t = await run_in_threadpool(_transcribe, transcriber, audio, await audio.read(), language)
    try:
        result = await run_in_threadpool(pipeline.run_text_query, llm, dataset_id, t.text, explain)
    except pipeline.PipelineError as e:                    # keep the transcript so the user sees what was heard
        e.context.update({"transcript": t.text, "language": t.language})
        raise
    return {"transcript": t.text, "language": t.language, **result}


# ----------------------------------------------------------------- UI
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
