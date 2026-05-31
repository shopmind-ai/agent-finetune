import json
import os
import psycopg2
from config import DATABASE_URL


def extract_training_data() -> list[dict]:
    """Extract Q&A pairs from PostgreSQL and format as MLX training samples."""
    samples: list[dict] = []
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT product_name, scenario, messages
                FROM training_sessions
                WHERE jsonb_array_length(messages) >= 2
                LIMIT 100
            """)
            for product_name, scenario, messages in cur.fetchall():
                msgs = messages if isinstance(messages, list) else json.loads(messages or "[]")
                user_msgs = [m for m in msgs if m.get("role") == "user"]
                asst_msgs = [m for m in msgs if m.get("role") == "assistant"]
                for u, a in zip(user_msgs, asst_msgs):
                    if u.get("content") and a.get("content"):
                        samples.append({
                            "text": f"Human: {u['content']}\n\nAssistant: {a['content']}"
                        })

            cur.execute("""
                SELECT messages
                FROM cs_sessions
                WHERE jsonb_array_length(messages) >= 2
                LIMIT 50
            """)
            for (messages,) in cur.fetchall():
                msgs = messages if isinstance(messages, list) else json.loads(messages or "[]")
                user_msgs = [m for m in msgs if m.get("role") == "user"]
                asst_msgs = [m for m in msgs if m.get("role") == "assistant"]
                for u, a in zip(user_msgs, asst_msgs):
                    if u.get("content") and a.get("content"):
                        samples.append({
                            "text": f"Human: {u['content']}\n\nAssistant: {a['content']}"
                        })
    finally:
        conn.close()
    return samples


def save_training_data(samples: list[dict],
                       output_path: str = "data/train.jsonl") -> int:
    """Write samples as JSONL. Returns number of samples written."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return len(samples)
