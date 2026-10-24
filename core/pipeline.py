"""End-to-end orchestration:  question text -> SQL -> validate -> SQLite -> result -> explanation."""
from __future__ import annotations

import json
import threading
from collections import defaultdict

from . import config
from .datasets import describe
from .explainer import explain
from .sql_generator import fix_sql, generate_sql
from .sql_runner import SQLExecutionError, SQLValidationError, execute_sql

_history: dict[str, list[dict]] = defaultdict(list)     # short per-dataset conversation memory
_hlock = threading.Lock()


class PipelineError(Exception):
    """A user-facing failure with an HTTP status and the partial context gathered so far."""

    def __init__(self, message: str, status: int = 422, context: dict | None = None):
        super().__init__(message)
        self.status, self.context = status, context or {}


def get_history(dataset_id: str) -> list[dict]:
    with _hlock:
        return list(_history[dataset_id][-config.HISTORY_TURNS:])


def clear_history(dataset_id: str) -> None:
    with _hlock:
        _history.pop(dataset_id, None)


def _remember(dataset_id: str, question: str, sql: str) -> None:
    with _hlock:
        _history[dataset_id].append({"question": question, "sql": sql})
        del _history[dataset_id][: -config.HISTORY_TURNS * 2]


def generate_valid_sql(llm, info, question: str, history=None, execute: bool = True):
    """Generate SQL, then validate+execute; on failure feed the error back to the LLM (self-correction).
    Returns (sql, QueryResult|None, attempts)."""
    out = generate_sql(llm, info, question, history)
    attempts = 0
    while True:
        if "error" in out:
            raise PipelineError(out["error"], 422)
        sql = out["sql"]
        try:
            res = execute_sql(info.dataset_id, sql) if execute else None
            if not execute:
                from .sql_runner import validate_sql
                sql = validate_sql(sql)
            return sql, res, attempts
        except (SQLValidationError, SQLExecutionError) as e:
            if attempts >= config.MAX_SQL_RETRIES:
                raise PipelineError(f"Could not produce a valid SQL query: {e}", 422, {"sql": sql})
            attempts += 1
            out = fix_sql(llm, info, question, sql, str(e), history)


def run_text_query(llm, dataset_id: str, question: str, with_explanation: bool = True) -> dict:
    question = (question or "").strip()
    if not question:
        raise PipelineError("Empty question.", 400)
    info = describe(dataset_id)
    history = get_history(dataset_id)
    sql, res, attempts = generate_valid_sql(llm, info, question, history)
    _remember(dataset_id, question, sql)
    explanation = explain(llm, question, sql, res) if with_explanation else None
    return {"question": question, "sql": sql, "columns": res.columns, "rows": res.rows,
            "row_count": res.row_count, "truncated": res.truncated, "elapsed_ms": res.elapsed_ms,
            "sql_retries": attempts, "explanation": explanation}
