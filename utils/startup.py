import psycopg2
from config import DATABASE_URL

_DDL = """
CREATE TABLE IF NOT EXISTS finetune_jobs (
    id                SERIAL PRIMARY KEY,
    job_id            VARCHAR(36) UNIQUE NOT NULL DEFAULT gen_random_uuid()::text,
    action            VARCHAR(30) NOT NULL,
    prompt            TEXT        DEFAULT '',
    base_response     TEXT        DEFAULT '',
    finetuned_response TEXT       DEFAULT '',
    data_count        INTEGER     DEFAULT 0,
    input_tokens      INTEGER     DEFAULT 0,
    output_tokens     INTEGER     DEFAULT 0,
    created_at        TIMESTAMPTZ DEFAULT NOW()
);
"""


def check_and_init():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set.")
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(_DDL)
        conn.commit()
        print("[startup] agent-finetune DB initialized.")
    finally:
        conn.close()


def log_job(action: str, prompt: str = "", base_response: str = "",
            finetuned_response: str = "", data_count: int = 0, usage: dict = None):
    usage = usage or {}
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO finetune_jobs (action, prompt, base_response, finetuned_response, "
                "data_count, input_tokens, output_tokens) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (action, prompt, base_response, finetuned_response, data_count,
                 usage.get("input_tokens", 0), usage.get("output_tokens", 0))
            )
        conn.commit()
    finally:
        conn.close()
