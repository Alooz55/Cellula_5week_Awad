"""Short natural-language explanation of a query result, in the user's language."""
from __future__ import annotations

import json

from .sql_generator import is_arabic
from .sql_runner import QueryResult

EXPLAIN_SYSTEM = """You explain database query results to a non-technical user in 2-3 short sentences.
Reply in the SAME language as the user's question (Arabic or English).
Use ONLY numbers that appear in the result rows; never invent or estimate values.
Mention the most important finding (top/bottom value, total, trend). No SQL, no markdown."""


def fallback_explanation(question: str, res: QueryResult) -> str:
    ar = is_arabic(question)
    if res.row_count == 0:
        return "لم يتم العثور على نتائج مطابقة لسؤالك." if ar else "No rows matched your question."
    if res.row_count == 1 and len(res.columns) == 1:
        v = res.rows[0][0]
        return f"النتيجة: {v}" if ar else f"The result is {v}."
    return (f"تم إرجاع {res.row_count} صف." if ar else f"The query returned {res.row_count} rows.")


def explain(llm, question: str, sql: str, res: QueryResult, max_rows: int = 20) -> str:
    if res.row_count == 0:
        return fallback_explanation(question, res)
    payload = {"question": question, "columns": res.columns, "rows": res.rows[:max_rows],
               "total_rows": res.row_count, "truncated": res.truncated}
    try:
        text = llm.chat([{"role": "system", "content": EXPLAIN_SYSTEM},
                         {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)}],
                        temperature=0.2)
        return text.strip() or fallback_explanation(question, res)
    except Exception:
        return fallback_explanation(question, res)
