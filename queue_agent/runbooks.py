import json
import os
from pathlib import Path

import psycopg
from langchain_ollama import OllamaEmbeddings
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parent.parent / "runbooks"
SOURCE_ROOT = "https://github.com/baimurat00747-hub/job-queue-with-ai-diagnostics/blob/main/runbooks"
SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS queue_runbook_sections (
    section_id TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    failure_code TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector NOT NULL,
    PRIMARY KEY (section_id, embedding_model)
);
CREATE INDEX IF NOT EXISTS queue_runbook_reason
ON queue_runbook_sections (failure_code, embedding_model);
"""


def ollama_url():
    configured = os.environ.get("QUEUE_AGENT_EMBEDDING_URL")
    if not configured and os.environ.get("QUEUE_AGENT_PROVIDER", "ollama") == "ollama":
        configured = os.environ.get("QUEUE_AGENT_MODEL_URL")
    return (configured or "http://127.0.0.1:11434").rstrip("/").removesuffix("/v1")


def database_url():
    return (
        os.environ.get("QUEUE_RUNBOOK_DATABASE_URL")
        or os.environ.get("QUEUE_DATABASE_URL")
        or "postgresql://queue:queue@127.0.0.1:55432/queue"
    )


def embeddings():
    return OllamaEmbeddings(
        model=os.environ.get("QUEUE_AGENT_EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
        base_url=ollama_url(),
        client_kwargs={"timeout": 90},
    )


def sections():
    for path in sorted(ROOT.glob("*.md")):
        _, *blocks = path.read_text().split("\n## ")
        for block in blocks:
            heading, _, content = block.partition("\n")
            anchor = heading.lower().replace(" ", "-")
            yield (
                f"{path.stem}/{anchor}",
                path.stem.upper(),
                f"{path.stem.upper()} — {heading}",
                f"{SOURCE_ROOT}/{path.name}#{anchor}",
                content.strip(),
            )


def seed():
    documents = list(sections())
    embedder = embeddings()
    vectors = embedder.embed_documents([f"{title}\n{content}" for _, _, title, _, content in documents])
    with psycopg.connect(database_url(), connect_timeout=5) as conn:
        conn.execute(SCHEMA)
        with conn.cursor() as cursor:
            cursor.executemany(
                """INSERT INTO queue_runbook_sections
                       (section_id, embedding_model, failure_code, title, url, content, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                   ON CONFLICT (section_id, embedding_model) DO UPDATE SET
                       title = EXCLUDED.title, url = EXCLUDED.url,
                       content = EXCLUDED.content, embedding = EXCLUDED.embedding""",
                [
                    (section_id, embedder.model, code, title, url, content, json.dumps(vector, allow_nan=False))
                    for (section_id, code, title, url, content), vector in zip(documents, vectors, strict=True)
                ],
            )
    return len(documents)


RETRIEVAL = """
WITH request AS (
    SELECT %s::vector AS vector, %s::text AS query
), terms AS (
    SELECT to_tsquery('english', string_agg(quote_literal(term), ' | ')) AS query
    FROM request, unnest(tsvector_to_array(to_tsvector('english', request.query))) AS term
), scored AS (
    SELECT section_id, title, url, content, 1 - (embedding <=> request.vector) AS similarity,
           COALESCE(ts_rank(to_tsvector('english', title || ' ' || content), terms.query), 0) AS lexical
    FROM queue_runbook_sections, request, terms
    WHERE failure_code = %s AND embedding_model = %s
      AND vector_dims(embedding) = vector_dims(request.vector)
), ranked AS (
    SELECT *, rank() OVER (ORDER BY similarity DESC) AS vector_rank,
           rank() OVER (ORDER BY lexical DESC) AS lexical_rank
    FROM scored
)
SELECT title, url, content, similarity, lexical,
       1.0 / (60 + vector_rank) + CASE WHEN %s AND lexical > 0
           THEN 1.0 / (60 + lexical_rank) ELSE 0 END AS rank
FROM ranked
WHERE similarity >= %s
ORDER BY rank DESC, similarity DESC, section_id LIMIT %s
"""


async def retrieve_runbooks(code: str, question: str, *, hybrid=True, limit=3):
    embedder = embeddings()
    vector = json.dumps(await embedder.aembed_query(question), allow_nan=False)
    async with await psycopg.AsyncConnection.connect(
        database_url(), connect_timeout=5, options="-c statement_timeout=5000", row_factory=dict_row,
    ) as conn:
        floor = float(os.environ.get("QUEUE_AGENT_MIN_SIMILARITY", "0.35"))
        rows = await conn.execute(RETRIEVAL, (vector, question, code, embedder.model, hybrid, floor, limit))
        return await rows.fetchall()


async def search_runbooks(code: str, question: str):
    return await retrieve_runbooks(code, question, limit=1)


if __name__ == "__main__":
    print(f"Indexed {seed()} runbook sections.")
