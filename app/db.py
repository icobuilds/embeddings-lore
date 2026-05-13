"""
Database access layer.

"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pymysql
import pymysql.cursors

from app.config import AppConfig

from logging import getLogger

logger = getLogger(__name__)
logger.info("Initializing database layer")

# ── Connection ────────────────────────────────────────────────────────────────

@contextmanager
def get_connection(cfg: AppConfig) -> Iterator[pymysql.connections.Connection]:
    conn = pymysql.connect(
        host=cfg.database.host,
        port=cfg.database.port,
        user=cfg.database.user,
        password=cfg.database.password,
        database=cfg.database.database,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
    )
    try:
        logger.info(f"Connected to database: {cfg.database.host}:{cfg.database.port}")
        logger.info(f"User: {cfg.database.user}")
        logger.info(f"Database: {cfg.database.database}")
        logger.info(f"Table: {cfg.database.table}")
        yield conn
    finally:
        conn.close()
        logger.info("Disconnected from database")


# ── Readers ───────────────────────────────────────────────────────────────────

def count_candidates(cfg: AppConfig) -> int:
    """Return total number of rows in the configured table."""
    logger.info(f"Counting candidates in table: {cfg.database.table}")
    with get_connection(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) AS n FROM {cfg.database.table}")
            row = cur.fetchone()
            logger.info(f"Total candidates: {row['n']}")
            return int(row["n"]) if row else 0


def iter_candidates(
    cfg: AppConfig,
    batch_size: int = 500,
) -> Iterator[list[dict[str, Any]]]:
    """
    Yield successive batches of raw row dicts from the editions table.
    Uses keyset pagination on candidate_id to avoid large OFFSETs.
    """
    last_isbn: str = ""
    table = cfg.database.table

    with get_connection(cfg) as conn:
        while True:
            with conn.cursor() as cur:
                if last_isbn:
                    cur.execute(
                        f"SELECT * FROM {table} WHERE {cfg.embedding.candidate_id} > %s "
                        f"ORDER BY {cfg.embedding.candidate_id} LIMIT %s",
                        (last_isbn, batch_size),
                    )
                else:
                    cur.execute(
                        f"SELECT * FROM {table} ORDER BY {cfg.embedding.candidate_id} LIMIT %s",
                        (batch_size,),
                    )
                rows = cur.fetchall()

            if not rows:
                break

            yield list(rows)
            last_isbn = rows[-1][cfg.embedding.candidate_id]


def load_all_candidates(cfg: AppConfig) -> tuple[list[str], list[str]]:
    """
    Return (candidate_id_list, text_list) for every book in the table.

    candidate_id_list[i]  — the candidate_id for candidate i  (used to map search results back)
    text_list[i]  — the embeddable text string for candidate i
    """
    logger.info(f"Loading all candidates from table: {cfg.database.table}")
    candidate_ids: list[str] = []
    texts: list[str] = []

    for batch in iter_candidates(cfg, batch_size=cfg.index.batch_size):
        for row in batch:
            candidate_id = row.get(cfg.embedding.candidate_id) or ""
            text = cfg.build_text(row)
            if candidate_id and text:
                candidate_ids.append(candidate_id)
                texts.append(text)

    return candidate_ids, texts


def build_text_for_query(cfg: AppConfig, fields: dict[str, Any]) -> str:
    """
    Build an embeddable text string from a user-supplied field dict
    (e.g., from the /similarity request body).  Missing fields fall back
    to empty string, matching the same template used for indexed books.
    """
    return cfg.build_text(fields)
