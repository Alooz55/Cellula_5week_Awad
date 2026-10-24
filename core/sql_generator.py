"""LLM text -> SQL generation (Arabic + English), with JSON parsing and self-correction."""
from __future__ import annotations

import json
import re

from .datasets import DatasetInfo, schema_prompt

SYSTEM_PROMPT = """You are an expert data analyst who writes SQLite SQL.
The user asks a question about ONE table, in Arabic or English (the question may come from speech
recognition, so expect small spelling mistakes). Translate it into a single SQLite query.

Rules:
- Output ONLY a JSON object: {"sql": "<query>"}  or, if the question cannot be answered from the table,
  {"error": "<short reason in the user's language>"}.
- Only a single read-only SELECT (or WITH ... SELECT). Never modify data.
- Use ONLY the table and columns listed in the schema. Copy column names exactly, wrapped in double quotes.
  The question may be Arabic while column names are English (or vice versa): map the meaning to the closest column.
- When filtering on text values, use the exact values listed in the schema hints when they exist.
- Give aggregates readable aliases (e.g. SUM("sales") AS total_sales). Use ORDER BY for rankings, LIMIT for "top N".
- For dates stored as text (YYYY-MM-DD) use strftime(). Use ROUND(x, 2) for averages/ratios.
- Do not invent columns. If several columns could match and it is truly ambiguous, pick the most likely one.

Examples (table "sales" with columns "category","amount","order_date","city"):
Q: Show me total sales by product category
{"sql": "SELECT \\"category\\", SUM(\\"amount\\") AS total_sales FROM \\"sales\\" GROUP BY \\"category\\" ORDER BY total_sales DESC"}
Q: اعرض إجمالي المبيعات لكل فئة
{"sql": "SELECT \\"category\\", SUM(\\"amount\\") AS total_sales FROM \\"sales\\" GROUP BY \\"category\\" ORDER BY total_sales DESC"}
Q: ما هي أعلى ٣ مدن من حيث عدد الطلبات؟
{"sql": "SELECT \\"city\\", COUNT(*) AS orders_count FROM \\"sales\\" GROUP BY \\"city\\" ORDER BY orders_count DESC LIMIT 3"}
Q: How many orders were placed in 2024?
{"sql": "SELECT COUNT(*) AS orders_2024 FROM \\"sales\\" WHERE strftime('%Y', \\"order_date\\") = '2024'"}
Q: كم متوسط قيمة الطلب في كل شهر؟
{"sql": "SELECT strftime('%Y-%m', \\"order_date\\") AS month, ROUND(AVG(\\"amount\\"), 2) AS avg_order FROM \\"sales\\" GROUP BY month ORDER BY month"}
"""

_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩٫٬", "0123456789.,")


def is_arabic(text: str) -> bool:
    return len(re.findall(r"[؀-ۿ]", text or "")) > len(re.findall(r"[A-Za-z]", text or ""))


def normalize_question(q: str) -> str:
    return (q or "").translate(_AR_DIGITS).strip()


def parse_llm_output(text: str) -> dict:
    """Accepts JSON, JSON in a code fence, a ```sql block, or bare SQL. Returns {'sql':..} or {'error':..}."""
    text = (text or "").strip()
    fence = re.search(r"```(?:json|sql)?\s*(.*?)```", text, re.S | re.I)
    body = fence.group(1).strip() if fence else text
    for candidate in (body, text):
        m = re.search(r"\{.*\}", candidate, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and (obj.get("sql") or obj.get("error")):
                    return {k: obj[k] for k in ("sql", "error") if obj.get(k)}
            except json.JSONDecodeError:
                pass
    m = re.search(r"(?is)\b(select|with)\b.*", body)
    if m:
        return {"sql": m.group(0).strip()}
    return {"error": "The model did not return a SQL query."}


def build_messages(info: DatasetInfo, question: str, history: list[dict] | None = None) -> list[dict]:
    ctx = "Database schema:\n" + schema_prompt(info)
    if history:
        ctx += "\n\nPrevious questions in this session (for follow-ups like 'and for last year?'):\n"
        ctx += "\n".join(f"- Q: {h['question']}\n  SQL: {h['sql']}" for h in history)
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{ctx}\n\nQuestion: {normalize_question(question)}"}]


def generate_sql(llm, info: DatasetInfo, question: str, history: list[dict] | None = None) -> dict:
    return parse_llm_output(llm.chat(build_messages(info, question, history)))


def fix_sql(llm, info: DatasetInfo, question: str, bad_sql: str, error: str,
            history: list[dict] | None = None) -> dict:
    """Self-correction: show the model its failing query and the error."""
    msgs = build_messages(info, question, history)
    msgs.append({"role": "assistant", "content": json.dumps({"sql": bad_sql}, ensure_ascii=False)})
    msgs.append({"role": "user", "content": f"That query failed with: {error}\n"
                                            "Fix it. Use only columns from the schema. Reply with the JSON object only."})
    return parse_llm_output(llm.chat(msgs))
