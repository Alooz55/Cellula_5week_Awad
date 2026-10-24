"""SQL validation + safe execution on SQLite3 (read-only, authorizer, row cap, timeout)."""
from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass

from . import config
from .datasets import db_path


class SQLValidationError(Exception):
    pass


class SQLExecutionError(Exception):
    pass


FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|pragma|vacuum|reindex|"
    r"analyze|begin|commit|rollback|savepoint|release|load_extension)\b", re.I)


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    return re.sub(r"--[^\n]*", " ", sql)


def _mask_strings(sql: str) -> str:
    """Replace quoted literals so keywords inside text values don't trigger the checks."""
    return re.sub(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"", "''", sql)


def validate_sql(sql: str) -> str:
    """Return the cleaned single SELECT/WITH statement or raise SQLValidationError."""
    if not sql or not sql.strip():
        raise SQLValidationError("Empty SQL query.")
    clean = _strip_comments(sql).strip()
    clean = re.sub(r";\s*$", "", clean).strip()
    masked = _mask_strings(clean)
    if ";" in masked:
        raise SQLValidationError("Only a single SQL statement is allowed.")
    if not re.match(r"(?is)^\s*(select|with)\b", masked):
        raise SQLValidationError("Only SELECT queries are allowed.")
    m = FORBIDDEN.search(masked)
    if m:
        raise SQLValidationError(f"Forbidden keyword in query: {m.group(1).upper()}")
    return clean


_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
if hasattr(sqlite3, "SQLITE_RECURSIVE"):
    _ALLOWED_ACTIONS.add(sqlite3.SQLITE_RECURSIVE)


def _authorizer(action, *_):
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[list]
    row_count: int
    truncated: bool
    elapsed_ms: float

    def to_dict(self) -> dict:
        return self.__dict__


def execute_sql(dataset_id: str, sql: str) -> QueryResult:
    clean = validate_sql(sql)
    path = db_path(dataset_id)
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=config.SQL_TIMEOUT_SEC)
    start = time.monotonic()
    try:
        # SQLite quirk: an unknown "double-quoted" identifier silently becomes a string literal, which would hide
        # hallucinated column names. Turn that off so they raise "no such column" and trigger self-correction.
        if hasattr(con, "setconfig"):                       # Python 3.12+
            con.setconfig(sqlite3.SQLITE_DBCONFIG_DQS_DML, False)
            con.setconfig(sqlite3.SQLITE_DBCONFIG_DQS_DDL, False)
        con.set_authorizer(_authorizer)
        con.set_progress_handler(lambda: 1 if time.monotonic() - start > config.SQL_TIMEOUT_SEC else 0, 10000)
        try:
            cur = con.execute(clean)
            rows = cur.fetchmany(config.MAX_ROWS + 1)
            columns = [d[0] for d in cur.description] if cur.description else []
        except sqlite3.Error as e:
            msg = "Query timed out." if "interrupted" in str(e).lower() else str(e)
            raise SQLExecutionError(msg)
    finally:
        con.close()
    truncated = len(rows) > config.MAX_ROWS
    rows = [list(r) for r in rows[: config.MAX_ROWS]]
    return QueryResult(columns, rows, len(rows), truncated, round((time.monotonic() - start) * 1000, 2))
