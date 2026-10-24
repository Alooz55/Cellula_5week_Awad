"""Offline end-to-end tests (mock LLM + fake speech-to-text). Run:  pytest -q"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from core import config, pipeline, stt
from core.sql_generator import parse_llm_output
from core.sql_runner import SQLValidationError, validate_sql
from main import app, llm_dep, transcriber_dep

DATA = Path(__file__).resolve().parent.parent / "data"


class MockLLM:
    """Scripted LLM: SQL requests pop from `sql_replies`; explanation requests return a fixed text."""

    def __init__(self, sql_replies):
        self.sql_replies, self.calls = list(sql_replies), []

    def chat(self, messages, temperature=0.0):
        self.calls.append(messages)
        if "explain database query results" in messages[0]["content"]:
            return "Electronics has the highest total."
        return self.sql_replies.pop(0)


class FakeSTT:
    def __init__(self, text="Show me total amount by category", lang="en"):
        self.text, self.lang = text, lang

    def __call__(self, data, filename, language):
        return stt.Transcript(self.text, language or self.lang, 0.99, 3.0)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STORAGE_DIR", tmp_path)
    pipeline._history.clear()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def upload(client, name="sales_sample.csv"):
    with open(DATA / name, "rb") as f:
        r = client.post("/datasets/upload", files={"file": (name, f, "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()


def use(llm=None, stt_fn=None):
    if llm:
        app.dependency_overrides[llm_dep] = lambda: llm
    if stt_fn:
        app.dependency_overrides[transcriber_dep] = lambda: stt_fn


GOOD = json.dumps({"sql": 'SELECT "category", SUM("amount") AS total FROM "data" GROUP BY "category" ORDER BY total DESC'})


# ---------------------------------------------------------------- datasets
def test_upload_and_schema(client):
    d = upload(client)
    assert d["row_count"] == 400 and d["table"] == "data"
    assert [c["name"] for c in d["columns"]][:3] == ["order_id", "order_date", "category"]
    cat = next(c for c in d["columns"] if c["name"] == "category")
    assert set(cat["examples"]) == {"Electronics", "Clothing", "Home & Kitchen", "Books"}
    assert client.get(f"/datasets/{d['dataset_id']}/schema").json()["row_count"] == 400


def test_upload_arabic_columns(client):
    d = upload(client, "مبيعات_عينة.csv")
    assert "رقم_الطلب" in [c["name"] for c in d["columns"]]      # space -> underscore, Arabic kept


def test_upload_rejects_bad_files(client):
    assert client.post("/datasets/upload", files={"file": ("x.exe", b"abc")}).status_code == 400
    assert client.post("/datasets/upload", files={"file": ("x.csv", b"")}).status_code == 400


def test_unknown_dataset(client):
    assert client.get("/datasets/aaaaaaaaaaaa/schema").status_code == 404
    assert client.get("/datasets/..%2f..%2fetc/schema").status_code in (404, 422)


# ---------------------------------------------------------------- SQL safety
@pytest.mark.parametrize("sql", [
    "DROP TABLE data", "DELETE FROM data", "UPDATE data SET amount=0", "SELECT 1; DROP TABLE data",
    "PRAGMA table_info(data)", "ATTACH DATABASE 'x.db' AS x", "INSERT INTO data VALUES (1)", "", "  ",
    "/* hi */ DROP TABLE data", "SELECT * FROM data; SELECT 2",
])
def test_validator_rejects(sql):
    with pytest.raises(SQLValidationError):
        validate_sql(sql)


def test_validator_accepts():
    assert validate_sql('SELECT * FROM "data";') == 'SELECT * FROM "data"'
    assert validate_sql("WITH t AS (SELECT 1 AS a) SELECT a FROM t")
    assert validate_sql("SELECT * FROM data WHERE city = 'drop; table'")      # keyword inside a string literal


def test_execute_endpoint(client):
    d = upload(client)
    r = client.post("/execute-sql", json={"dataset_id": d["dataset_id"], "sql": 'SELECT COUNT(*) AS n FROM "data"'})
    assert r.status_code == 200 and r.json()["rows"] == [[400]]
    bad = client.post("/execute-sql", json={"dataset_id": d["dataset_id"], "sql": "DROP TABLE data"})
    assert bad.status_code == 400
    assert client.post("/execute-sql", json={"dataset_id": d["dataset_id"], "sql": "SELECT nope FROM data"}).status_code == 422
    # data survived
    assert client.get(f"/datasets/{d['dataset_id']}/schema").json()["row_count"] == 400


def test_row_cap(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_ROWS", 10)
    d = upload(client)
    r = client.post("/execute-sql", json={"dataset_id": d["dataset_id"], "sql": 'SELECT * FROM "data"'}).json()
    assert r["row_count"] == 10 and r["truncated"] is True


# ---------------------------------------------------------------- LLM parsing
@pytest.mark.parametrize("raw,key", [
    ('{"sql": "SELECT 1"}', "sql"),
    ('```json\n{"sql": "SELECT 1"}\n```', "sql"),
    ("```sql\nSELECT 1\n```", "sql"),
    ("Here you go: SELECT 1 FROM data", "sql"),
    ('{"error": "no such info"}', "error"),
    ("I cannot help", "error"),
])
def test_parse_llm_output(raw, key):
    assert key in parse_llm_output(raw)


# ---------------------------------------------------------------- pipeline
def test_text_query_english(client):
    d = upload(client)
    use(MockLLM([GOOD]))
    r = client.post("/text-query", json={"dataset_id": d["dataset_id"], "question": "total amount by category"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["columns"] == ["category", "total"] and j["row_count"] == 4
    assert j["sql_retries"] == 0 and j["explanation"]


def test_text_query_arabic_prompt_contains_schema(client):
    d = upload(client)
    llm = MockLLM([GOOD])
    use(llm)
    r = client.post("/text-query", json={"dataset_id": d["dataset_id"], "question": "اعرض إجمالي المبيعات لكل فئة ٣"})
    assert r.status_code == 200
    user_msg = llm.calls[0][1]["content"]
    assert '"category" TEXT' in user_msg and "اعرض إجمالي" in user_msg and "٣" not in user_msg   # Arabic digits normalised


def test_self_correction(client):
    d = upload(client)
    llm = MockLLM([json.dumps({"sql": 'SELECT SUM("sales") FROM "data"'}), GOOD])    # 1st attempt: hallucinated column
    use(llm)
    j = client.post("/text-query", json={"dataset_id": d["dataset_id"], "question": "total by category"}).json()
    assert j["sql_retries"] == 1 and j["row_count"] == 4
    assert "no such column" in llm.calls[1][-1]["content"]


def test_unsafe_sql_from_llm_is_blocked_then_fails(client):
    d = upload(client)
    use(MockLLM([json.dumps({"sql": "DROP TABLE data"})] * 3))
    r = client.post("/text-query", json={"dataset_id": d["dataset_id"], "question": "delete everything"})
    assert r.status_code == 422
    assert client.get(f"/datasets/{d['dataset_id']}/schema").json()["row_count"] == 400


def test_llm_declines(client):
    d = upload(client)
    use(MockLLM([json.dumps({"error": "لا يوجد عمود مناسب"})]))
    r = client.post("/text-query", json={"dataset_id": d["dataset_id"], "question": "ما هو الطقس؟"})
    assert r.status_code == 422 and "لا يوجد" in r.json()["detail"]


def test_conversation_memory(client):
    d = upload(client)
    llm = MockLLM([GOOD, GOOD])
    use(llm)
    for q in ("total by category", "and for Amman only?"):
        client.post("/text-query", json={"dataset_id": d["dataset_id"], "question": q})
    assert "total by category" in llm.calls[-2][1]["content"] or "total by category" in str(llm.calls)
    assert client.delete(f"/datasets/{d['dataset_id']}/history").json()["cleared"]


def test_generate_sql_endpoint(client):
    d = upload(client)
    use(MockLLM([GOOD]))
    r = client.post("/generate-sql", json={"dataset_id": d["dataset_id"], "question": "x"})
    assert r.status_code == 200 and r.json()["sql"].startswith("SELECT")


# ---------------------------------------------------------------- voice
def test_transcribe_endpoint(client):
    use(stt_fn=FakeSTT("اعرض إجمالي المبيعات", "ar"))
    r = client.post("/transcribe", files={"audio": ("a.wav", b"RIFFfake")})
    assert r.status_code == 200 and r.json()["language"] == "ar"


def test_voice_query_full_pipeline_arabic(client):
    d = upload(client)
    use(MockLLM([GOOD]), FakeSTT("اعرض إجمالي المبلغ لكل فئة", "ar"))
    r = client.post("/voice-query", data={"dataset_id": d["dataset_id"]}, files={"audio": ("a.webm", b"fake-bytes")})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["transcript"].startswith("اعرض") and j["language"] == "ar"
    assert j["sql"].startswith("SELECT") and j["row_count"] == 4 and j["explanation"]


def test_voice_errors(client):
    d = upload(client)
    use(MockLLM([GOOD]), FakeSTT(""))
    assert client.post("/voice-query", data={"dataset_id": d["dataset_id"]}, files={"audio": ("a.wav", b"x")}).status_code == 422
    assert client.post("/voice-query", data={"dataset_id": d["dataset_id"]}, files={"audio": ("a.wav", b"")}).status_code == 400
    assert client.post("/voice-query", data={"dataset_id": "aaaaaaaaaaaa"}, files={"audio": ("a.wav", b"x")}).status_code == 404


def test_health_and_ui(client):
    assert client.get("/health").json()["status"] == "ok"
    assert "Voice-Based Data Analysis" in client.get("/").text
