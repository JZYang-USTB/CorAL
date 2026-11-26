from __future__ import annotations

import hashlib
import re
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

from pypdf import PdfReader

from .schemas import SourceEvidence


DOMAIN_EXPANSIONS = {
    "corrosion": ["corrosion inhibitor", "inhibitor", "inhibition"],
    "synergy": ["synergistic", "synergism", "cooperative"],
    "mixture": ["combination", "combined", "cocktail"],
    "formulation": ["mixture", "cocktail", "composition"],
    "electrochemical": ["impedance", "polarization", "EIS"],
}


class RagStore:
    def __init__(self, database: Path):
        self.database = database
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            self._initialize(connection)

    @contextmanager
    def _connect(self):
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute("PRAGMA journal_mode=WAL")
            yield connection

    @staticmethod
    def _initialize(connection: sqlite3.Connection):
        connection.execute(
            "CREATE TABLE IF NOT EXISTS documents("
            "id TEXT PRIMARY KEY, path TEXT NOT NULL, size INTEGER, mtime_ns INTEGER)"
        )
        connection.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5("
            "document_id UNINDEXED, chunk_number UNINDEXED, content, "
            "tokenize='unicode61')"
        )

    @staticmethod
    def _extract_text(path: Path) -> str:
        pages = []
        for page in PdfReader(str(path)).pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages)

    @staticmethod
    def _split(text: str, size: int = 1800, overlap: int = 200):
        normalized = re.sub(r"\s+", " ", text).strip()
        step = max(1, size - overlap)
        for start in range(0, len(normalized), step):
            chunk = normalized[start : start + size]
            yield chunk
            if start + size >= len(normalized):
                break

    def ingest(self, corpus: Path, limit: int | None = None) -> dict:
        pdfs = sorted(corpus.rglob("*.pdf"))
        if limit is not None:
            pdfs = pdfs[:limit]
        indexed = skipped = failed = chunks = 0
        errors = []
        with self._connect() as connection:
            for path in pdfs:
                try:
                    stat = path.stat()
                    with path.open("rb") as stream:
                        document_id = hashlib.file_digest(stream, "sha256").hexdigest()
                    if connection.execute(
                        "SELECT 1 FROM documents WHERE id=?", (document_id,)
                    ).fetchone():
                        skipped += 1
                        continue
                    text = self._extract_text(path)
                    rows = [
                        (document_id, number, chunk)
                        for number, chunk in enumerate(self._split(text))
                    ]
                    if not rows:
                        raise ValueError("No extractable text.")
                    connection.executemany(
                        "INSERT INTO chunks(document_id,chunk_number,content) "
                        "VALUES(?,?,?)",
                        rows,
                    )
                    connection.execute(
                        "INSERT INTO documents(id,path,size,mtime_ns) VALUES(?,?,?,?)",
                        (document_id, path.relative_to(corpus).as_posix(), stat.st_size, stat.st_mtime_ns),
                    )
                    connection.commit()
                    indexed += 1
                    chunks += len(rows)
                except Exception as error:
                    connection.rollback()
                    failed += 1
                    errors.append({"path": str(path), "error": str(error)})
        return {
            "discovered": len(pdfs),
            "indexed": indexed,
            "skipped": skipped,
            "failed": failed,
            "chunks_added": chunks,
            "errors": errors[:100],
        }

    @staticmethod
    def _query_terms(question: str) -> list[str]:
        query = question.casefold()
        terms = re.findall(r"[a-z0-9]{2,}", query)
        terms = [term for term in terms if term not in {"and", "or", "not", "the", "of", "for", "in", "to", "with"}]
        expanded = [term for key, related in DOMAIN_EXPANSIONS.items() if key in query for term in related]
        return list(dict.fromkeys(expanded + terms))[:24]

    def search(self, question: str, top_k: int = 8) -> list[SourceEvidence]:
        terms = self._query_terms(question)
        fts_query = " OR ".join(f'"{term}"' for term in terms) or '"corrosion inhibitor"'
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT d.id,d.path,c.chunk_number,c.content,bm25(chunks) "
                "FROM chunks c JOIN documents d ON d.id=c.document_id "
                "WHERE chunks MATCH ? ORDER BY bm25(chunks),d.id,c.chunk_number LIMIT ?",
                (fts_query, top_k),
            ).fetchall()
        return [
            SourceEvidence(
                source_id=f"D-{document_id}-C{chunk_number:04d}",
                document_path=path,
                chunk_number=chunk_number,
                content=content,
                score=score,
            )
            for document_id, path, chunk_number, content, score in rows
        ]

    def status(self) -> dict:
        with self._connect() as connection:
            documents = connection.execute("SELECT count(*) FROM documents").fetchone()[0]
            chunks = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
        return {"database": str(self.database), "documents": documents, "chunks": chunks}

