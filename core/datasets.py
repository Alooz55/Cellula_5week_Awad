"""Dataset ingestion: CSV / TSV / Excel / JSON  ->  one SQLite database per dataset (table name: data)."""
from __future__ import annotations

import io
import json
import re
import sqlite3
import uuid
from dataclasses import dataclass, field

import pandas as pd

from . import config

TABLE = "data"
ALLOWED_EXT = (".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json")


class DatasetError(Exception):
    """User-facing dataset problem (maps to HTTP 400/404)."""


class DatasetNotFound(DatasetError):
    pass


@dataclass
class DatasetInfo:
    dataset_id: str
    filename: str
    table: str
    row_count: int
    columns: list[dict]                      # [{name, type, distinct, examples}]
    sample: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__


def db_path(dataset_id: str):
    if not re.fullmatch(r"[0-9a-f]{12}", dataset_id or ""):
        raise DatasetNotFound(f"Invalid dataset_id '{dataset_id}'")
    p = config.STORAGE_DIR / f"{dataset_id}.db"
    if not p.exists():
        raise DatasetNotFound(f"Dataset '{dataset_id}' not found. Upload it first via POST /datasets/upload.")
    return p


def clean_columns(cols) -> list[str]:
    """Trim, replace spaces/symbols with '_', keep Arabic + Latin letters, make unique & non-empty."""
    out, seen = [], {}
    for i, c in enumerate(cols):
        name = re.sub(r"\W+", "_", str(c).strip(), flags=re.UNICODE).strip("_") or f"col_{i + 1}"
        if name[0].isdigit():
            name = f"c_{name}"
        base, n = name, seen.get(name.lower(), 0)
        if n:
            name = f"{base}_{n + 1}"
        seen[base.lower()] = n + 1
        out.append(name)
    return out


def _read_csv(raw: bytes, sep) -> pd.DataFrame:
    last = None
    for enc in ("utf-8-sig", "cp1256", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc, engine="python")
        except UnicodeDecodeError as e:
            last = e
    raise DatasetError(f"Could not decode the file: {last}")


def read_dataframe(filename: str, raw: bytes) -> pd.DataFrame:
    name = (filename or "").lower()
    if not name.endswith(ALLOWED_EXT):
        raise DatasetError(f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXT)}")
    try:
        if name.endswith(".tsv"):
            df = _read_csv(raw, "\t")
        elif name.endswith((".csv", ".txt")):
            df = _read_csv(raw, None)          # None + python engine = sniff the delimiter
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(raw))
        else:
            df = pd.read_json(io.BytesIO(raw))
    except DatasetError:
        raise
    except Exception as e:
        raise DatasetError(f"Could not parse '{filename}': {e}")
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty or len(df.columns) == 0:
        raise DatasetError("The file has no data rows.")
    return df


def _sql_type(dtype) -> str:
    if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    if pd.api.types.is_float_dtype(dtype):
        return "REAL"
    return "TEXT"


def create_dataset(filename: str, raw: bytes) -> DatasetInfo:
    if len(raw) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise DatasetError(f"File too large (max {config.MAX_UPLOAD_MB} MB).")
    df = read_dataframe(filename, raw)
    df.columns = clean_columns(df.columns)
    for c in df.columns:                                   # try to parse date-like text columns
        if df[c].dtype == object:
            s = df[c].dropna().astype(str)
            if len(s) and s.str.fullmatch(r"\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?").all():
                df[c] = pd.to_datetime(df[c], errors="coerce").dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%d")
    dataset_id = uuid.uuid4().hex[:12]
    path = config.STORAGE_DIR / f"{dataset_id}.db"
    con = sqlite3.connect(path)
    try:
        df.to_sql(TABLE, con, index=False, if_exists="replace")
        con.commit()
    finally:
        con.close()
    meta = describe(dataset_id, filename)
    (config.STORAGE_DIR / f"{dataset_id}.json").write_text(json.dumps(meta.to_dict(), ensure_ascii=False, default=str),
                                                           encoding="utf-8")
    return meta


def describe(dataset_id: str, filename: str = "") -> DatasetInfo:
    """Read schema / stats back from the SQLite file (read-only)."""
    path = db_path(dataset_id)
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        cols_info = con.execute(f'PRAGMA table_info("{TABLE}")').fetchall()
        n = con.execute(f'SELECT COUNT(*) FROM "{TABLE}"').fetchone()[0]
        cols = []
        for _, name, typ, *_ in cols_info:
            distinct = con.execute(f'SELECT COUNT(DISTINCT "{name}") FROM "{TABLE}"').fetchone()[0]
            ex_n = 15 if (typ == "TEXT" and distinct <= 15) else 3
            examples = [r[0] for r in con.execute(
                f'SELECT DISTINCT "{name}" FROM "{TABLE}" WHERE "{name}" IS NOT NULL LIMIT {ex_n}')]
            cols.append({"name": name, "type": typ or "TEXT", "distinct": distinct, "examples": examples})
        cur = con.execute(f'SELECT * FROM "{TABLE}" LIMIT 5')
        names = [d[0] for d in cur.description]
        sample = [dict(zip(names, r)) for r in cur.fetchall()]
    finally:
        con.close()
    if not filename:
        meta = config.STORAGE_DIR / f"{dataset_id}.json"
        if meta.exists():
            filename = json.loads(meta.read_text(encoding="utf-8")).get("filename", "")
    return DatasetInfo(dataset_id, filename, TABLE, n, cols, sample)


def schema_prompt(info: DatasetInfo) -> str:
    """Compact schema description for the LLM: columns, types, value hints, sample rows."""
    lines = [f'Table: "{info.table}"  ({info.row_count} rows)', "Columns:"]
    for c in info.columns:
        hint = ""
        if c["type"] == "TEXT" and c["distinct"] <= 15:
            hint = f"  -- all distinct values: {c['examples']}"
        elif c["examples"]:
            hint = f"  -- e.g. {c['examples'][:3]}"
        lines.append(f'  "{c["name"]}" {c["type"]}{hint}')
    lines.append("Sample rows:")
    for r in info.sample[:3]:
        lines.append("  " + json.dumps(r, ensure_ascii=False, default=str))
    return "\n".join(lines)
