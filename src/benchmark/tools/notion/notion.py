"""Local Notion benchmark wrappers backed by a baseline and task overlay."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import re
import sqlite3
import threading
import tomllib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, Callable

import hnswlib
import numpy as np

# FastEmbed otherwise may follow Hugging Face downloads through several Xet
# hosts. Use the regular download path to reduce network overhead.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from fastembed import TextEmbedding


BENCHMARK_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_CONFIG_PATH = BENCHMARK_DIR / "config.toml"
DEFAULT_RUNTIME_DIR = BENCHMARK_DIR / ".runtime" / "notion"


def _load_data_directories() -> tuple[Path, Path]:
    with BENCHMARK_CONFIG_PATH.open("rb") as config_file:
        config = tomllib.load(config_file)
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ValueError(f"Missing [paths] table in {BENCHMARK_CONFIG_PATH}")

    configured: list[Path] = []
    for key in ("data_dir", "task_overlay_data_dir"):
        value = paths.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing paths.{key} in {BENCHMARK_CONFIG_PATH}")
        path = Path(value)
        configured.append(path if path.is_absolute() else BENCHMARK_DIR / path)
    return configured[0], configured[1]


DATA_DIR, TASK_OVERLAY_DATA_DIR = _load_data_directories()
DEFAULT_BASELINE_DIR = DATA_DIR / "notion"
DEFAULT_OVERLAY_DIR = TASK_OVERLAY_DATA_DIR / "notion"


class NotionToolError(ValueError):
    """A predictable, agent-readable tool error."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def as_dict(self) -> dict[str, Any]:
        return {"object": "error", "code": self.code, "message": self.message}


@dataclass(frozen=True)
class NotionConfig:
    """All paths, limits, chunking, embedding, and vector-index settings."""

    baseline_dir: Path = DEFAULT_BASELINE_DIR
    overlay_dir: Path = DEFAULT_OVERLAY_DIR
    runtime_dir: Path = DEFAULT_RUNTIME_DIR
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: Path | None = None
    embedding_model_path: Path | None = None
    embedding_threads: int | None = None
    chunk_characters: int = 1_400
    chunk_overlap: int = 180
    embed_batch_size: int = 128
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 96
    hnsw_random_seed: int = 17
    hnsw_threads: int = -1
    index_build_batch_size: int = 10_000
    search_oversample: int = 8
    highlight_characters: int = 360
    max_page_size: int = 100
    sqlite_timeout_seconds: float = 30.0
    embedding_local_files_only: bool = False
    overlay_runtime_dir: Path | None = None

    @classmethod
    def from_env(cls) -> "NotionConfig":
        cache = os.getenv("NOTION_EMBEDDING_CACHE")
        threads = os.getenv("NOTION_EMBEDDING_THREADS")
        return cls(
            baseline_dir=Path(
                os.getenv("BENCHMARK_NOTION_BASELINE_DIR", DEFAULT_BASELINE_DIR)
            ),
            overlay_dir=Path(
                os.getenv("BENCHMARK_NOTION_OVERLAY_DIR", DEFAULT_OVERLAY_DIR)
            ),
            runtime_dir=Path(
                os.getenv("BENCHMARK_NOTION_RUNTIME_DIR", DEFAULT_RUNTIME_DIR)
            ),
            embedding_model=os.getenv("NOTION_EMBEDDING_MODEL", cls.embedding_model),
            embedding_cache_dir=Path(cache) if cache else None,
            embedding_threads=int(threads) if threads else None,
            chunk_characters=int(
                os.getenv("NOTION_CHUNK_CHARACTERS", cls.chunk_characters)
            ),
            chunk_overlap=int(os.getenv("NOTION_CHUNK_OVERLAP", cls.chunk_overlap)),
            embed_batch_size=int(
                os.getenv("NOTION_EMBED_BATCH_SIZE", cls.embed_batch_size)
            ),
            hnsw_m=int(os.getenv("NOTION_HNSW_M", cls.hnsw_m)),
            hnsw_ef_construction=int(
                os.getenv("NOTION_HNSW_EF_CONSTRUCTION", cls.hnsw_ef_construction)
            ),
            hnsw_ef_search=int(os.getenv("NOTION_HNSW_EF_SEARCH", cls.hnsw_ef_search)),
            hnsw_random_seed=int(
                os.getenv("NOTION_HNSW_RANDOM_SEED", cls.hnsw_random_seed)
            ),
            hnsw_threads=int(os.getenv("NOTION_HNSW_THREADS", cls.hnsw_threads)),
            index_build_batch_size=int(
                os.getenv("NOTION_INDEX_BUILD_BATCH_SIZE", cls.index_build_batch_size)
            ),
            search_oversample=int(
                os.getenv("NOTION_SEARCH_OVERSAMPLE", cls.search_oversample)
            ),
            highlight_characters=int(
                os.getenv("NOTION_HIGHLIGHT_CHARACTERS", cls.highlight_characters)
            ),
            max_page_size=int(os.getenv("NOTION_MAX_PAGE_SIZE", cls.max_page_size)),
            sqlite_timeout_seconds=float(
                os.getenv("NOTION_SQLITE_TIMEOUT_SECONDS", cls.sqlite_timeout_seconds)
            ),
        )

    def validate(self) -> None:
        if self.chunk_characters < 1:
            raise ValueError("chunk_characters must be positive")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_characters:
            raise ValueError("chunk_overlap must be between zero and chunk_characters")
        for name in (
            "embed_batch_size",
            "hnsw_m",
            "hnsw_ef_construction",
            "hnsw_ef_search",
            "index_build_batch_size",
            "search_oversample",
            "highlight_characters",
            "max_page_size",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS databases (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    parent_page_id TEXT,
    properties_json TEXT NOT NULL DEFAULT '{}',
    created_time TEXT NOT NULL,
    last_edited_time TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    parent_page_id TEXT,
    database_id TEXT,
    content_path TEXT,
    properties_json TEXT NOT NULL DEFAULT '{}',
    created_time TEXT NOT NULL,
    last_edited_time TEXT NOT NULL,
    archived INTEGER NOT NULL DEFAULT 0,
    operation TEXT NOT NULL DEFAULT 'upsert' CHECK (operation IN ('upsert', 'delete'))
);

CREATE INDEX IF NOT EXISTS pages_parent_idx ON pages(parent_page_id, archived);
CREATE INDEX IF NOT EXISTS pages_database_idx ON pages(database_id, archived);

CREATE TABLE IF NOT EXISTS page_properties (
    page_id TEXT NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    value_text TEXT,
    value_number REAL,
    value_boolean INTEGER,
    value_time TEXT,
    PRIMARY KEY (page_id, name, kind, value_text, value_number, value_boolean, value_time)
);

CREATE INDEX IF NOT EXISTS properties_text_idx
    ON page_properties(name, value_text, page_id);
CREATE INDEX IF NOT EXISTS properties_number_idx
    ON page_properties(name, value_number, page_id);
CREATE INDEX IF NOT EXISTS properties_boolean_idx
    ON page_properties(name, value_boolean, page_id);
CREATE INDEX IF NOT EXISTS properties_time_idx
    ON page_properties(name, value_time, page_id);

CREATE TABLE IF NOT EXISTS blocks (
    id TEXT PRIMARY KEY,
    page_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS blocks_page_idx ON blocks(page_id, position);

CREATE TABLE IF NOT EXISTS search_chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id TEXT NOT NULL,
    block_id TEXT,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS search_chunks_page_idx ON search_chunks(page_id);
"""


EFFECTIVE_CTES = """
WITH effective_pages AS (
    SELECT p.*, 'overlay' AS source
    FROM main.pages p
    WHERE p.operation = 'upsert'
    UNION ALL
    SELECT p.*, 'baseline' AS source
    FROM baseline.pages p
    WHERE p.operation = 'upsert'
      AND NOT EXISTS (SELECT 1 FROM main.pages o WHERE o.id = p.id)
),
effective_properties AS (
    SELECT pp.*
    FROM main.page_properties pp
    JOIN main.pages p ON p.id = pp.page_id AND p.operation = 'upsert'
    UNION ALL
    SELECT pp.*
    FROM baseline.page_properties pp
    WHERE NOT EXISTS (SELECT 1 FROM main.pages o WHERE o.id = pp.page_id)
)
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _safe_relative_path(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    resolved_root = root.resolve()
    if target != resolved_root and resolved_root not in target.parents:
        raise NotionToolError(
            "invalid_path", f"Page path leaves its data directory: {relative}"
        )
    return target


def _markdown_blocks(content: str) -> list[tuple[str, str]]:
    paragraphs = [
        part.strip() for part in re.split(r"\n\s*\n", content) if part.strip()
    ]
    blocks: list[tuple[str, str]] = []
    for paragraph in paragraphs:
        first_line = paragraph.splitlines()[0]
        kind = "heading" if re.match(r"^#{1,6}\s+", first_line) else "paragraph"
        blocks.append((kind, paragraph))
    return blocks or [("paragraph", "")]


def _chunk_text(title: str, content: str, config: NotionConfig) -> list[str]:
    prefix = f"Title: {title}\n\n"
    text = content.strip()
    if not text:
        return [prefix.strip()]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + config.chunk_characters)
        if end < len(text):
            boundary = text.rfind("\n\n", start, end)
            if boundary <= start:
                boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        body = text[start:end].strip()
        if body:
            chunks.append(prefix + body)
        if end >= len(text):
            break
        next_start = max(start + 1, end - config.chunk_overlap)
        start = next_start
    return chunks


class _Embeddings:
    def __init__(self, config: NotionConfig):
        self.config = config
        self._model: TextEmbedding | None = None
        self._lock = threading.Lock()

    def _get_model(self) -> TextEmbedding:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        self._model = TextEmbedding(
                            model_name=self.config.embedding_model,
                            cache_dir=(
                                str(self.config.embedding_cache_dir)
                                if self.config.embedding_cache_dir
                                else None
                            ),
                            threads=self.config.embedding_threads,
                            local_files_only=self.config.embedding_local_files_only,
                            **(
                                {
                                    "specific_model_path": str(
                                        self.config.embedding_model_path
                                    )
                                }
                                if self.config.embedding_model_path is not None
                                else {}
                            ),
                        )
                    except Exception as error:
                        if self.config.embedding_local_files_only:
                            raise NotionToolError(
                                "embedding_unavailable",
                                "Cached Notion embedding model is unavailable. Prepare the configured model cache.",
                            ) from error
                        raise
        return self._model

    def documents(self, texts: Sequence[str]) -> list[np.ndarray[Any, Any]]:
        if not texts:
            return []
        vectors = self._get_model().embed(
            texts, batch_size=self.config.embed_batch_size
        )
        return [np.asarray(vector, dtype=np.float32) for vector in vectors]

    def query(self, text: str) -> np.ndarray[Any, Any]:
        return np.asarray(
            next(iter(self._get_model().query_embed(text))), dtype=np.float32
        )


def _property_rows(
    page_id: str, properties: Mapping[str, Any]
) -> Iterator[tuple[str, str, str, str | None, float | None, int | None, str | None]]:
    for name, property_value in properties.items():
        if not isinstance(property_value, Mapping):
            yield (page_id, name, "text", str(property_value), None, None, None)
            continue

        kind = str(property_value.get("type") or _infer_property_kind(property_value))
        raw = property_value.get(kind)
        values: list[Any]
        if kind in {"multi_select", "people", "relation"}:
            values = list(raw or [])
        else:
            values = [raw]

        if not values:
            yield (page_id, name, kind, None, None, None, None)
            continue

        for value in values:
            text: str | None = None
            number: float | None = None
            boolean: int | None = None
            time: str | None = None

            if kind == "number" and value is not None:
                number = float(value)
            elif kind == "checkbox" and value is not None:
                boolean = int(bool(value))
            elif kind == "date" and isinstance(value, Mapping):
                time = value.get("start")
            elif isinstance(value, Mapping):
                candidate = value.get("name", value.get("id", value.get("plain_text")))
                text = str(candidate) if candidate is not None else None
            elif value is not None:
                text = str(value)

            yield (page_id, name, kind, text, number, boolean, time)


def _infer_property_kind(value: Mapping[str, Any]) -> str:
    known = (
        "title",
        "rich_text",
        "number",
        "select",
        "status",
        "multi_select",
        "date",
        "people",
        "files",
        "checkbox",
        "url",
        "email",
        "phone_number",
        "relation",
    )
    return next((kind for kind in known if kind in value), "rich_text")


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA_SQL)


def _set_metadata(connection: sqlite3.Connection, key: str, value: Any) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", (key, str(value))
    )


def _insert_page_without_vectors(
    connection: sqlite3.Connection,
    page: Mapping[str, Any],
    content: str,
    config: NotionConfig,
) -> list[tuple[str, str]]:
    page_id = str(page["id"])
    operation = str(page.get("operation", "upsert"))
    properties = dict(page.get("properties", {}))
    connection.execute("DELETE FROM page_properties WHERE page_id = ?", (page_id,))
    connection.execute("DELETE FROM blocks WHERE page_id = ?", (page_id,))
    connection.execute("DELETE FROM search_chunks WHERE page_id = ?", (page_id,))
    connection.execute(
        """
        INSERT OR REPLACE INTO pages(
            id, title, parent_page_id, database_id, content_path, properties_json,
            created_time, last_edited_time, archived, operation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            page_id,
            str(page.get("title", "")),
            page.get("parent_page_id"),
            page.get("database_id"),
            page.get("content_path"),
            _json(properties),
            str(page.get("created_time", _utc_now())),
            str(page.get("last_edited_time", page.get("created_time", _utc_now()))),
            int(bool(page.get("archived", operation == "delete"))),
            operation,
        ),
    )
    if operation == "delete":
        return []

    connection.executemany(
        """
        INSERT INTO page_properties(
            page_id, name, kind, value_text, value_number, value_boolean, value_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        _property_rows(page_id, properties),
    )
    blocks = _markdown_blocks(content)
    for position, (kind, text) in enumerate(blocks):
        connection.execute(
            "INSERT INTO blocks(id, page_id, position, kind, text) VALUES (?, ?, ?, ?, ?)",
            (f"{page_id}:block:{position}", page_id, position, kind, text),
        )
    return [
        (page_id, text)
        for text in _chunk_text(str(page.get("title", "")), content, config)
    ]


def build_overlay_catalog(
    seed_path: Path,
    output_path: Path,
    data_root: Path,
    config: NotionConfig,
    *,
    read_only: bool,
    index_content: bool = True,
) -> None:
    """Compile task pages into a searchable overlay in one embedding pass."""

    config.validate()
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()

    connection = sqlite3.connect(temporary)
    connection.row_factory = sqlite3.Row
    chunks: list[tuple[str, str]] = []
    try:
        _create_schema(connection)
        for database in seed.get("databases", []):
            now = str(database.get("created_time", _utc_now()))
            connection.execute(
                """
                INSERT INTO databases(
                    id, title, parent_page_id, properties_json, created_time, last_edited_time
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(database["id"]),
                    str(database["title"]),
                    database.get("parent_page_id"),
                    _json(database.get("properties", {})),
                    now,
                    str(database.get("last_edited_time", now)),
                ),
            )

        for page in seed.get("pages", []):
            content_path = page.get("content_path")
            content = ""
            if (
                index_content
                and content_path
                and page.get("operation", "upsert") != "delete"
            ):
                content = _safe_relative_path(data_root, str(content_path)).read_text(
                    encoding="utf-8"
                )
            chunks.extend(
                _insert_page_without_vectors(connection, page, content, config)
            )

        if not index_content:
            chunks = []
        embedder = _Embeddings(config)
        dimension = 0
        for start in range(0, len(chunks), config.embed_batch_size):
            batch = chunks[start : start + config.embed_batch_size]
            embeddings = embedder.documents([text for _, text in batch])
            if embeddings:
                dimension = int(embeddings[0].shape[0])
            connection.executemany(
                "INSERT INTO search_chunks(page_id, block_id, text, embedding) VALUES (?, NULL, ?, ?)",
                [
                    (page_id, text, vector.tobytes())
                    for (page_id, text), vector in zip(batch, embeddings, strict=True)
                ],
            )

        _set_metadata(connection, "embedding_model", config.embedding_model)
        _set_metadata(connection, "embedding_dimension", dimension)
        _set_metadata(connection, "content_revision", 1)
        _set_metadata(connection, "seed_file", seed_path.name)
        connection.commit()
    finally:
        connection.close()

    os.replace(temporary, output_path)
    output_path.chmod(0o444 if read_only else 0o644)


@dataclass
class _LoadedIndex:
    signature: str
    index: hnswlib.Index
    count: int
    dimension: int


class NotionTools:
    """Five Notion wrappers over one immutable baseline and one writable overlay."""

    def __init__(
        self,
        config: NotionConfig | None = None,
        *,
        acquire_file: Callable | None = None,
        file_inventory: Callable | None = None,
    ):
        self.config = config or NotionConfig.from_env()
        self.config.validate()
        self.baseline_db = self.config.baseline_dir / "catalog.sqlite"
        self.overlay_db = self.config.overlay_dir / "overlay.sqlite"
        if not self.baseline_db.exists():
            raise FileNotFoundError(
                f"Missing Notion baseline catalog: {self.baseline_db}"
            )
        if not self.overlay_db.exists():
            raise FileNotFoundError(
                f"Missing Notion overlay catalog: {self.overlay_db}"
            )

        self.config.runtime_dir.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self.overlay_db,
            timeout=self.config.sqlite_timeout_seconds,
            check_same_thread=False,
            uri=True,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode = WAL")
        busy_timeout_ms = int(self.config.sqlite_timeout_seconds * 1_000)
        self._connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
        baseline_uri = f"file:{self.baseline_db.resolve()}?mode=ro&immutable=1"
        self._connection.execute("ATTACH DATABASE ? AS baseline", (baseline_uri,))
        self._lock = threading.RLock()
        self._embeddings = _Embeddings(self.config)
        self._indexes: dict[str, _LoadedIndex] = {}
        self.acquire_file = acquire_file
        self.file_inventory = file_inventory

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "NotionTools":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def notion_search(
        self,
        query: str,
        *,
        page_size: int = 10,
        page_id: str | None = None,
        database_id: str | None = None,
    ) -> dict[str, Any]:
        """Run semantic vector search over effective page content."""

        query = query.strip()
        if not query:
            raise NotionToolError("invalid_request", "query cannot be empty")
        page_size = self._page_size(page_size)
        query_vector = self._embeddings.query(query)
        candidates: dict[str, dict[str, Any]] = {}

        with self._lock:
            for source in ("overlay", "baseline"):
                for candidate in self._search_source(source, query_vector, page_size):
                    effective = self._effective_page(candidate["page_id"])
                    if (
                        effective is None
                        or effective["source"] != source
                        or effective["archived"]
                    ):
                        continue
                    if database_id and effective["database_id"] != database_id:
                        continue
                    if page_id and not self._is_within_page(effective["id"], page_id):
                        continue
                    current = candidates.get(candidate["page_id"])
                    if current is None or candidate["score"] > current["score"]:
                        candidates[candidate["page_id"]] = candidate

            ordered = sorted(
                candidates.values(), key=lambda item: (-item["score"], item["page_id"])
            )
            results = []
            for candidate in ordered[:page_size]:
                page_row = self._effective_page(candidate["page_id"])
                if page_row is None:
                    continue
                page = self._page_from_row(page_row, include_content=False)
                results.append(
                    {
                        "id": page["id"],
                        "title": page["title"],
                        "url": page["url"],
                        "parent": page["parent"],
                        "score": round(candidate["score"], 6),
                        "highlight": candidate["text"][
                            : self.config.highlight_characters
                        ],
                    }
                )
        return {
            "object": "list",
            "type": "page",
            "results": results,
            "result_scope": "ranked_subset",
            "complete": False,
            "discovery_tool": "notion_list_catalog",
        }

    def notion_fetch_page(self, page_id: str) -> dict[str, Any]:
        """Fetch one effective page, including its Markdown body."""

        normalized = _normalize_id(page_id)
        with self._lock:
            page_row = self._effective_page(normalized)
        if page_row is None or page_row["archived"]:
            raise NotionToolError("not_found", f"Page not found: {page_id}")
        return self._page_from_row(page_row)

    def _read_path(self, source: str, relative: str) -> Path:
        root = (
            self.config.overlay_dir if source == "overlay" else self.config.baseline_dir
        )
        path = _safe_relative_path(root, relative)
        if self.acquire_file is not None:
            path = self.acquire_file(source, relative)
        return path

    def read_body(
        self,
        source: str,
        relative: str,
        identity: str,
        *,
        start_cursor=None,
        max_chars=16000,
    ) -> dict[str, Any]:
        """Read a bounded UTF-8 slice. Cursor offsets refer to bytes, never task origins."""
        path = self._read_path(source, relative)
        offset = 0
        if start_cursor:
            cursor = self._decode_cursor(start_cursor)
            if (
                cursor.get("identity") != identity
                or type(cursor.get("offset")) is not int
                or cursor["offset"] < 0
            ):
                raise NotionToolError(
                    "invalid_cursor", "Cursor does not match this content"
                )
            offset = cursor["offset"]
        size = path.stat().st_size
        if offset > size:
            raise NotionToolError("invalid_cursor", "Cursor is past this content")
        with path.open("rb") as stream:
            stream.seek(offset)
            raw = stream.read(max_chars)
        # A byte cap is at least as strict as the requested character cap.
        while raw:
            try:
                content = raw.decode("utf-8")
                break
            except UnicodeDecodeError as error:
                if error.end != len(raw) or error.reason != "unexpected end of data":
                    raise NotionToolError(
                        "invalid_content", "Stored content is not UTF-8"
                    ) from error
                raw = raw[: error.start]
        else:
            content = ""
        end = offset + len(raw)
        more = end < size
        return dict(
            content=content,
            content_bytes=size,
            has_more=more,
            next_cursor=(
                self._encode_cursor(dict(identity=identity, offset=end))
                if more
                else None
            ),
        )

    def notion_list_catalog(
        self, *, kind="page", parent_page_id=None, page_size=50, start_cursor=None
    ) -> dict[str, Any]:
        """Metadata and hierarchy only. No body access or embedding calls."""
        page_size = self._page_size(page_size)
        if kind not in ("page", "database"):
            raise NotionToolError("invalid_request", "kind must be page or database")
        parent_page_id = _normalize_id(parent_page_id) if parent_page_id else None
        after = ""
        identity = dict(kind=kind, parent_page_id=parent_page_id)
        if start_cursor:
            cursor = self._decode_cursor(start_cursor)
            if cursor.get("catalog") != identity or not isinstance(
                cursor.get("id"), str
            ):
                raise NotionToolError(
                    "invalid_cursor", "Cursor does not match this catalog query"
                )
            after = cursor["id"]
        if kind == "page":
            sql = (
                EFFECTIVE_CTES
                + " SELECT * FROM effective_pages WHERE archived=0 AND id>?"
            )
        else:
            sql = """WITH effective_databases AS (
                SELECT * FROM main.databases UNION ALL SELECT * FROM baseline.databases b
                WHERE NOT EXISTS(SELECT 1 FROM main.databases o WHERE o.id=b.id))
                SELECT * FROM effective_databases WHERE id>?"""
        params = [after]
        if parent_page_id:
            sql += " AND parent_page_id=?"
            params.append(parent_page_id)
        rows = self._connection.execute(
            sql + " ORDER BY id LIMIT ?", [*params, page_size + 1]
        ).fetchall()
        more = len(rows) > page_size
        rows = rows[:page_size]
        if kind == "page":
            results = [self._page_from_row(r, include_content=False) for r in rows]
        else:
            results = [
                dict(
                    object="database",
                    id=r["id"],
                    title=r["title"],
                    parent={"page_id": r["parent_page_id"]},
                    properties=json.loads(r["properties_json"]),
                    created_time=r["created_time"],
                    last_edited_time=r["last_edited_time"],
                )
                for r in rows
            ]
        return dict(
            object="list",
            results=results,
            has_more=more,
            next_cursor=(
                self._encode_cursor(dict(catalog=identity, id=rows[-1]["id"]))
                if more
                else None
            ),
        )

    def history_rows(self, page_id: str, view="effective"):
        """Stream stored rows. Replace by native revision ID before paging."""
        replacements = {}
        overlay = self.config.overlay_dir / "revisions.jsonl"
        if view != "baseline" and overlay.is_file():
            with overlay.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        row = json.loads(line)
                        replacements[row["id"]] = row
        paths = (
            self.file_inventory("baseline")
            if self.file_inventory
            else [
                p.relative_to(self.config.baseline_dir).as_posix()
                for p in self.config.baseline_dir.rglob("*")
                if p.is_file()
            ]
        )
        if "revisions.jsonl" in paths:
            path = self._read_path("baseline", "revisions.jsonl")
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row = replacements.pop(row.get("id"), row)
                    if page_id is None or row.get("page_id") == page_id:
                        yield row
        for row in replacements.values():
            if page_id is None or row.get("page_id") == page_id:
                yield row

    def notion_list_history(
        self, page_id: str, *, kind="revisions", page_size=50, start_cursor=None
    ) -> dict[str, Any]:
        page_id = _normalize_id(page_id)
        if self._effective_page(page_id) is None:
            raise NotionToolError("not_found", "Page not found")
        page_size = self._page_size(page_size)
        identity = dict(page_id=page_id, kind=kind)
        offset = 0
        if start_cursor:
            cursor = self._decode_cursor(start_cursor)
            if (
                cursor.get("history") != identity
                or type(cursor.get("offset")) is not int
                or cursor["offset"] < 0
            ):
                raise NotionToolError(
                    "invalid_cursor", "Cursor does not match this history"
                )
            offset = cursor["offset"]
        if kind == "revisions":
            rows = self.history_rows(page_id)
        elif kind == "versions":
            paths = set()
            for source, root in (
                ("baseline", self.config.baseline_dir),
                ("overlay", self.config.overlay_dir),
            ):
                paths.update(
                    self.file_inventory(source)
                    if self.file_inventory
                    else [
                        p.relative_to(root).as_posix()
                        for p in root.rglob("*")
                        if p.is_file()
                    ]
                )
            prefix = f"versions/{page_id}/"
            rows = (
                dict(page_id=page_id, version=p[len(prefix) : -3])
                for p in sorted(paths)
                if p.startswith(prefix)
                and re.fullmatch(r"[A-Za-z0-9_-]+\.md", p[len(prefix) :])
            )
        else:
            raise NotionToolError(
                "invalid_request", "kind must be revisions or versions"
            )
        import itertools

        selected = list(itertools.islice(rows, offset, offset + page_size + 1))
        more = len(selected) > page_size
        return dict(
            object="list",
            results=selected[:page_size],
            has_more=more,
            next_cursor=(
                self._encode_cursor(dict(history=identity, offset=offset + page_size))
                if more
                else None
            ),
            stored_only=True,
        )

    def notion_fetch_version(
        self, page_id: str, version: str, *, start_cursor=None, max_chars=16000
    ) -> dict[str, Any]:
        page_id = _normalize_id(page_id)
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", version):
            raise NotionToolError("invalid_request", "Invalid version")
        if self._effective_page(page_id) is None:
            raise NotionToolError("not_found", "Page not found")
        relative = f"versions/{page_id}/{version}.md"
        source = (
            "overlay" if (self.config.overlay_dir / relative).is_file() else "baseline"
        )
        if self.file_inventory is not None and relative not in self.file_inventory(
            source
        ):
            raise NotionToolError("not_found", "Stored version not found")
        if (
            self.file_inventory is None
            and not (
                (
                    self.config.overlay_dir
                    if source == "overlay"
                    else self.config.baseline_dir
                )
                / relative
            ).is_file()
        ):
            raise NotionToolError("not_found", "Stored version not found")
        return dict(
            object="page_version",
            page_id=page_id,
            version=version,
            **self.read_body(
                source,
                relative,
                f"{page_id}/{version}",
                start_cursor=start_cursor,
                max_chars=max_chars,
            ),
        )

    def notion_query_database(
        self,
        database_id: str,
        *,
        filter: Mapping[str, Any] | None = None,
        sorts: Sequence[Mapping[str, Any]] | None = None,
        page_size: int = 100,
        start_cursor: str | None = None,
    ) -> dict[str, Any]:
        """Query effective database rows with indexed filters and keyset pagination."""

        database_id = _normalize_id(database_id)
        page_size = self._page_size(page_size)
        sorts_list = [dict(item) for item in (sorts or [])]
        with self._lock:
            if not self._database_exists(database_id):
                raise NotionToolError("not_found", f"Database not found: {database_id}")
            filter_params: list[Any] = []
            filter_sql = ""
            if filter:
                filter_sql = " AND " + self._compile_filter(filter, filter_params)

            sort_expressions: list[tuple[str, str]] = []
            sort_params: list[Any] = []
            for index, sort in enumerate(sorts_list):
                expression, expression_params, direction = self._sort_expression(sort)
                sort_params.extend(expression_params)
                sort_expressions.append((f"sort_{index}", direction))
                if index == 0:
                    select_sorts = f", {expression} AS sort_{index}"
                else:
                    select_sorts += f", {expression} AS sort_{index}"
            if not sort_expressions:
                select_sorts = ""

            cursor_sql = ""
            cursor_params: list[Any] = []
            cursor_payload = self._decode_cursor(start_cursor) if start_cursor else None
            if cursor_payload:
                if (
                    cursor_payload.get("database_id") != database_id
                    or cursor_payload.get("sorts") != sorts_list
                    or cursor_payload.get("filter") != filter
                ):
                    raise NotionToolError(
                        "invalid_cursor", "Cursor does not match this database query"
                    )
                cursor_sql, cursor_params = self._cursor_condition(
                    sort_expressions,
                    list(cursor_payload["keys"]),
                    str(cursor_payload["id"]),
                )

            order_parts = [
                f"COALESCE({alias}, '') {direction.upper()}"
                for alias, direction in sort_expressions
            ]
            order_parts.append("id ASC")
            sql = f"""
                {EFFECTIVE_CTES},
                filtered AS (
                    SELECT ep.* {select_sorts}
                    FROM effective_pages ep
                    WHERE ep.database_id = ? AND ep.archived = 0 {filter_sql}
                )
                SELECT * FROM filtered
                {cursor_sql}
                ORDER BY {", ".join(order_parts)}
                LIMIT ?
            """
            rows = self._connection.execute(
                sql,
                [
                    *sort_params,
                    database_id,
                    *filter_params,
                    *cursor_params,
                    page_size + 1,
                ],
            ).fetchall()
            has_more = len(rows) > page_size
            rows = rows[:page_size]
            results = [self._page_from_row(row, include_content=False) for row in rows]
            next_cursor = None
            if has_more and rows:
                last = rows[-1]
                next_cursor = self._encode_cursor(
                    {
                        "database_id": database_id,
                        "sorts": sorts_list,
                        "filter": filter,
                        "keys": [last[alias] for alias, _ in sort_expressions],
                        "id": last["id"],
                    }
                )
        return {
            "object": "list",
            "type": "page",
            "results": results,
            "has_more": has_more,
            "next_cursor": next_cursor,
        }

    def notion_create_page(
        self,
        parent: Mapping[str, str],
        title: str,
        *,
        content: str = "",
        properties: Mapping[str, Any] | None = None,
        page_id: str | None = None,
    ) -> dict[str, Any]:
        """Create one page or database row in the current task overlay."""

        title = title.strip()
        if not title:
            raise NotionToolError("invalid_request", "title cannot be empty")
        parent_page_id, database_id = self._validate_parent(parent)
        new_id = _normalize_id(page_id) if page_id else str(uuid.uuid4())
        with self._lock:
            if self._effective_page(new_id, include_archived=True) is not None:
                raise NotionToolError("conflict", f"Page already exists: {new_id}")
            now = _utc_now()
            page = {
                "id": new_id,
                "title": title,
                "parent_page_id": parent_page_id,
                "database_id": database_id,
                "content_path": f"pages/{new_id}.md",
                "properties": dict(properties or {}),
                "created_time": now,
                "last_edited_time": now,
                "archived": False,
                "operation": "upsert",
            }
            self._write_overlay_page(page, content)
            result_row = self._effective_page(new_id)
        assert result_row is not None
        return self._page_from_row(result_row)

    def notion_update_page(
        self,
        page_id: str,
        *,
        title: str | None = None,
        content: str | None = None,
        properties: Mapping[str, Any] | None = None,
        archived: bool | None = None,
    ) -> dict[str, Any]:
        """Copy an effective page into the overlay and apply the supplied changes."""

        page_id = _normalize_id(page_id)
        with self._lock:
            existing_row = self._effective_page(page_id, include_archived=True)
            if existing_row is None:
                raise NotionToolError("not_found", f"Page not found: {page_id}")
            existing = self._page_from_row(existing_row, include_content=True)
            if title is not None and not title.strip():
                raise NotionToolError("invalid_request", "title cannot be empty")
            merged_properties = dict(existing["properties"])
            if properties:
                merged_properties.update(properties)
            page = {
                "id": page_id,
                "title": title.strip() if title is not None else existing["title"],
                "parent_page_id": existing["parent"].get("page_id"),
                "database_id": existing["parent"].get("database_id"),
                "content_path": f"pages/{page_id}.md",
                "properties": merged_properties,
                "created_time": existing["created_time"],
                "last_edited_time": _utc_now(),
                "archived": bool(archived) if archived is not None else False,
                "operation": "delete" if archived else "upsert",
            }
            body = existing["content"] if content is None else content
            self._write_overlay_page(page, body)
            if page["operation"] == "delete":
                return {
                    "object": "page",
                    "id": page_id,
                    "archived": True,
                    "last_edited_time": page["last_edited_time"],
                }
            result_row = self._effective_page(page_id)
        assert result_row is not None
        return self._page_from_row(result_row)

    def _page_size(self, page_size: int) -> int:
        if (
            not isinstance(page_size, int)
            or page_size < 1
            or page_size > self.config.max_page_size
        ):
            raise NotionToolError(
                "invalid_request",
                f"page_size must be between 1 and {self.config.max_page_size}",
            )
        return page_size

    def _database_exists(self, database_id: str) -> bool:
        row = self._connection.execute(
            """
            SELECT 1 FROM main.databases WHERE id = ?
            UNION ALL
            SELECT 1 FROM baseline.databases WHERE id = ?
              AND NOT EXISTS (SELECT 1 FROM main.databases WHERE id = ?)
            LIMIT 1
            """,
            (database_id, database_id, database_id),
        ).fetchone()
        return row is not None

    def _validate_parent(
        self, parent: Mapping[str, str]
    ) -> tuple[str | None, str | None]:
        page_id = parent.get("page_id")
        database_id = parent.get("database_id")
        if bool(page_id) == bool(database_id):
            raise NotionToolError(
                "invalid_request",
                "parent must contain exactly one of page_id or database_id",
            )
        with self._lock:
            if page_id:
                normalized = _normalize_id(page_id)
                row = self._effective_page(normalized)
                if row is None or row["archived"]:
                    raise NotionToolError(
                        "not_found", f"Parent page not found: {page_id}"
                    )
                return normalized, None
            normalized = _normalize_id(str(database_id))
            if not self._database_exists(normalized):
                raise NotionToolError(
                    "not_found", f"Parent database not found: {database_id}"
                )
            return None, normalized

    def _effective_page(
        self, page_id: str, *, include_archived: bool = False
    ) -> sqlite3.Row | None:
        row = self._connection.execute(
            f"""
            {EFFECTIVE_CTES}
            SELECT * FROM effective_pages WHERE id = ?
            """,
            (page_id,),
        ).fetchone()
        if row is None:
            return None
        if not include_archived and row["archived"]:
            return None
        return row

    def _page_from_row(
        self, row: sqlite3.Row, *, include_content: bool = True
    ) -> dict[str, Any]:
        source = str(row["source"])
        parent = (
            {"database_id": row["database_id"]}
            if row["database_id"]
            else {"page_id": row["parent_page_id"]}
        )
        result: dict[str, Any] = {
            "object": "page",
            "id": row["id"],
            "url": f"notion://page/{row['id']}",
            "title": row["title"],
            "parent": parent,
            "properties": json.loads(row["properties_json"]),
            "created_time": row["created_time"],
            "last_edited_time": row["last_edited_time"],
            "archived": bool(row["archived"]),
        }
        if include_content:
            path = row["content_path"]
            result["content"] = (
                self._read_path(source, path).read_text(encoding="utf-8")
                if path
                else ""
            )
        return result

    def _is_within_page(self, candidate_id: str, ancestor_id: str) -> bool:
        ancestor_id = _normalize_id(ancestor_id)
        current = candidate_id
        visited: set[str] = set()
        while current and current not in visited:
            if current == ancestor_id:
                return True
            visited.add(current)
            row = self._effective_page(current, include_archived=True)
            if row is None:
                return False
            current = row["parent_page_id"]
        return False

    def _write_overlay_page(self, page: Mapping[str, Any], content: str) -> None:
        content_path = str(page["content_path"])
        page_path = _safe_relative_path(self.config.overlay_dir, content_path)
        page_path.parent.mkdir(parents=True, exist_ok=True)
        if page.get("operation", "upsert") != "delete":
            temporary = page_path.with_name(f".{page_path.name}.{os.getpid()}.tmp")
            temporary.write_text(content, encoding="utf-8")
            os.replace(temporary, page_path)

        chunks = (
            []
            if page.get("operation") == "delete"
            else _chunk_text(str(page["title"]), content, self.config)
        )
        vectors = self._embeddings.documents(chunks)
        with self._connection:
            _insert_page_without_vectors(self._connection, page, content, self.config)
            for text, vector in zip(chunks, vectors, strict=True):
                self._connection.execute(
                    "INSERT INTO search_chunks(page_id, block_id, text, embedding) VALUES (?, NULL, ?, ?)",
                    (page["id"], text, vector.tobytes()),
                )
            dimension = (
                int(vectors[0].shape[0])
                if vectors
                else self._metadata_int("main", "embedding_dimension")
            )
            _set_metadata(
                self._connection, "embedding_model", self.config.embedding_model
            )
            _set_metadata(self._connection, "embedding_dimension", dimension)
            _set_metadata(
                self._connection,
                "content_revision",
                self._metadata_int("main", "content_revision") + 1,
            )
        self._indexes.pop("overlay", None)

    def _search_source(
        self, source: str, query_vector: np.ndarray[Any, Any], page_size: int
    ) -> list[dict[str, Any]]:
        loaded = self._load_index(source)
        if loaded is None or loaded.count == 0:
            return []
        # A cosine query with a shorter vector can corrupt hnswlib's native
        # process memory. Check the catalog width before crossing that boundary.
        if query_vector.ndim != 1 or query_vector.size != loaded.dimension:
            raise NotionToolError(
                "index_dimension_mismatch",
                f"Query embedding dimension must match the catalog ({loaded.dimension}).",
            )
        target = min(
            loaded.count, max(page_size, page_size * self.config.search_oversample)
        )
        best: dict[str, dict[str, Any]] = {}
        schema = "main" if source == "overlay" else "baseline"
        while True:
            labels, distances = loaded.index.knn_query(query_vector, k=target)
            label_values = [int(label) for label in labels[0]]
            placeholders = ",".join("?" for _ in label_values)
            rows = self._connection.execute(
                f"SELECT chunk_id, page_id, text FROM {schema}.search_chunks WHERE chunk_id IN ({placeholders})",
                label_values,
            ).fetchall()
            by_label = {int(row["chunk_id"]): row for row in rows}
            for label, distance in zip(label_values, distances[0], strict=True):
                row = by_label.get(label)
                if row is None:
                    continue
                item = {
                    "page_id": row["page_id"],
                    "text": row["text"],
                    "score": 1.0 - float(distance),
                }
                current = best.get(item["page_id"])
                if current is None or item["score"] > current["score"]:
                    best[item["page_id"]] = item
            if len(best) >= page_size or target >= loaded.count:
                break
            target = min(loaded.count, target * 2)
        return list(best.values())

    def _load_index(self, source: str) -> _LoadedIndex | None:
        schema = "main" if source == "overlay" else "baseline"
        count = int(
            self._connection.execute(
                f"SELECT COUNT(*) FROM {schema}.search_chunks"
            ).fetchone()[0]
        )
        if count == 0:
            return None
        model = self._metadata(schema, "embedding_model")
        dimension = self._metadata_int(schema, "embedding_dimension")
        revision = self._metadata_int(schema, "content_revision")
        if model != self.config.embedding_model:
            raise NotionToolError(
                "index_model_mismatch",
                f"Catalog uses {model}, but NOTION_EMBEDDING_MODEL is {self.config.embedding_model}. Rebuild the fixture.",
            )
        signature_data = {
            "database": str(
                (self.overlay_db if source == "overlay" else self.baseline_db).resolve()
            ),
            "model": model,
            "dimension": dimension,
            "revision": revision,
            "count": count,
            "m": self.config.hnsw_m,
            "ef_construction": self.config.hnsw_ef_construction,
            "seed": self.config.hnsw_random_seed,
        }
        if source == "baseline":
            from benchmark.runtime.shared_source import selected_file_hash

            signature_data["catalog_sha256"] = selected_file_hash(
                self.baseline_db.resolve()
            )
        signature = hashlib.sha256(_json(signature_data).encode()).hexdigest()
        cached = self._indexes.get(source)
        if cached and cached.signature == signature:
            return cached

        stem = hashlib.sha256(signature_data["database"].encode()).hexdigest()[:16]
        runtime_dir = (
            self.config.overlay_runtime_dir
            if source == "overlay" and self.config.overlay_runtime_dir
            else self.config.runtime_dir
        )
        runtime_dir.mkdir(parents=True, exist_ok=True)
        index_path = runtime_dir / f"{stem}-{source}.hnsw"
        metadata_path = runtime_dir / f"{stem}-{source}.json"
        lock_path = runtime_dir / f"{stem}-{source}.lock"
        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            valid = False
            if index_path.exists() and metadata_path.exists():
                try:
                    valid = (
                        json.loads(metadata_path.read_text(encoding="utf-8")).get(
                            "signature"
                        )
                        == signature
                    )
                except (json.JSONDecodeError, OSError):
                    valid = False
            if not valid:
                self._build_index(schema, dimension, count, index_path)
                temporary_meta = metadata_path.with_name(
                    f".{metadata_path.name}.{os.getpid()}.tmp"
                )
                temporary_meta.write_text(
                    _json({"signature": signature, **signature_data}), encoding="utf-8"
                )
                os.replace(temporary_meta, metadata_path)

        index = hnswlib.Index(space="cosine", dim=dimension)
        index.load_index(str(index_path), max_elements=count)
        index.set_ef(max(self.config.hnsw_ef_search, 1))
        index.set_num_threads(self.config.hnsw_threads)
        loaded = _LoadedIndex(
            signature=signature, index=index, count=count, dimension=dimension
        )
        self._indexes[source] = loaded
        return loaded

    def _build_index(self, schema: str, dimension: int, count: int, path: Path) -> None:
        rows = self._connection.execute(
            f"SELECT chunk_id, embedding FROM {schema}.search_chunks ORDER BY chunk_id"
        )
        index = hnswlib.Index(space="cosine", dim=dimension)
        index.init_index(
            max_elements=count,
            ef_construction=self.config.hnsw_ef_construction,
            M=self.config.hnsw_m,
            random_seed=self.config.hnsw_random_seed,
        )
        index.set_num_threads(self.config.hnsw_threads)
        while batch := rows.fetchmany(self.config.index_build_batch_size):
            labels: list[int] = []
            vectors: list[np.ndarray[Any, Any]] = []
            for row in batch:
                labels.append(int(row["chunk_id"]))
                vector = np.frombuffer(row["embedding"], dtype=np.float32)
                if vector.size != dimension:
                    raise NotionToolError(
                        "invalid_index", "Stored embedding dimension is inconsistent"
                    )
                vectors.append(vector)
            index.add_items(np.vstack(vectors), np.asarray(labels, dtype=np.int64))
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        index.save_index(str(temporary))
        os.replace(temporary, path)

    def _metadata(self, schema: str, key: str) -> str:
        row = self._connection.execute(
            f"SELECT value FROM {schema}.metadata WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            raise NotionToolError("invalid_catalog", f"Missing catalog metadata: {key}")
        return str(row[0])

    def _metadata_int(self, schema: str, key: str) -> int:
        return int(self._metadata(schema, key))

    def _compile_filter(self, node: Mapping[str, Any], params: list[Any]) -> str:
        if "and" in node or "or" in node:
            operator = "and" if "and" in node else "or"
            children = node[operator]
            if (
                not isinstance(children, Sequence)
                or isinstance(children, (str, bytes))
                or not children
            ):
                raise NotionToolError(
                    "invalid_filter", f"{operator} must contain filters"
                )
            parts = [self._compile_filter(child, params) for child in children]
            return "(" + f" {operator.upper()} ".join(parts) + ")"

        if "timestamp" in node:
            timestamp = str(node["timestamp"])
            if timestamp not in {"created_time", "last_edited_time"}:
                raise NotionToolError(
                    "invalid_filter", f"Unsupported timestamp: {timestamp}"
                )
            condition = node.get(timestamp)
            if not isinstance(condition, Mapping):
                raise NotionToolError(
                    "invalid_filter", f"Missing {timestamp} condition"
                )
            return self._compile_scalar_condition(
                f"ep.{timestamp}", condition, params, "date"
            )

        property_name = node.get("property")
        if not property_name:
            raise NotionToolError("invalid_filter", "A property filter needs property")
        condition_kind = next((key for key in node if key != "property"), None)
        if condition_kind is None or not isinstance(node[condition_kind], Mapping):
            raise NotionToolError(
                "invalid_filter", f"Missing condition for {property_name}"
            )
        condition = node[condition_kind]
        field = _condition_field(str(condition_kind))
        clauses: list[str] = []
        for operator, value in condition.items():
            params.append(str(property_name))
            name_parameter = "?"
            if operator == "is_empty":
                clauses.append(
                    f"{'NOT ' if value else ''}EXISTS (SELECT 1 FROM effective_properties p WHERE p.page_id = ep.id AND p.name = {name_parameter} AND {field} IS NOT NULL)"
                )
                continue
            if operator == "is_not_empty":
                clauses.append(
                    f"{'NOT ' if not value else ''}EXISTS (SELECT 1 FROM effective_properties p WHERE p.page_id = ep.id AND p.name = {name_parameter} AND {field} IS NOT NULL)"
                )
                continue
            condition_sql, condition_params, negate = _property_condition(
                field, operator, value
            )
            params.extend(condition_params)
            clauses.append(
                f"{'NOT ' if negate else ''}EXISTS (SELECT 1 FROM effective_properties p WHERE p.page_id = ep.id AND p.name = {name_parameter} AND {condition_sql})"
            )
        return "(" + " AND ".join(clauses) + ")"

    def _compile_scalar_condition(
        self,
        expression: str,
        condition: Mapping[str, Any],
        params: list[Any],
        kind: str,
    ) -> str:
        parts = []
        for operator, value in condition.items():
            sql, values, negate = _property_condition(
                expression, operator, value, direct=True
            )
            params.extend(values)
            parts.append(f"{'NOT ' if negate else ''}({sql})")
        if not parts:
            raise NotionToolError("invalid_filter", f"Empty {kind} condition")
        return "(" + " AND ".join(parts) + ")"

    def _sort_expression(self, sort: Mapping[str, Any]) -> tuple[str, list[Any], str]:
        direction = str(sort.get("direction", "ascending")).lower()
        if direction not in {"ascending", "descending"}:
            raise NotionToolError(
                "invalid_sort", f"Unsupported sort direction: {direction}"
            )
        sql_direction = "asc" if direction == "ascending" else "desc"
        if "timestamp" in sort:
            timestamp = str(sort["timestamp"])
            if timestamp not in {"created_time", "last_edited_time"}:
                raise NotionToolError(
                    "invalid_sort", f"Unsupported timestamp: {timestamp}"
                )
            return f"ep.{timestamp}", [], sql_direction
        if "property" in sort:
            return (
                "(SELECT COALESCE(p.value_number, p.value_time, p.value_text, p.value_boolean) "
                "FROM effective_properties p WHERE p.page_id = ep.id AND p.name = ? LIMIT 1)",
                [str(sort["property"])],
                sql_direction,
            )
        raise NotionToolError("invalid_sort", "Sort needs property or timestamp")

    def _cursor_condition(
        self,
        sorts: Sequence[tuple[str, str]],
        keys: list[Any],
        page_id: str,
    ) -> tuple[str, list[Any]]:
        if len(keys) != len(sorts):
            raise NotionToolError("invalid_cursor", "Cursor sort values are invalid")
        aliases = [alias for alias, _ in sorts] + ["id"]
        directions = [direction for _, direction in sorts] + ["asc"]
        values = keys + [page_id]
        branches: list[str] = []
        params: list[Any] = []
        for index, (alias, direction, value) in enumerate(
            zip(aliases, directions, values, strict=True)
        ):
            equals = []
            for prior_alias, prior_value in zip(
                aliases[:index], values[:index], strict=True
            ):
                equals.append(f"COALESCE({prior_alias}, '') = COALESCE(?, '')")
                params.append(prior_value)
            comparison = ">" if direction == "asc" else "<"
            params.append(value)
            branch = " AND ".join(
                [*equals, f"COALESCE({alias}, '') {comparison} COALESCE(?, '')"]
            )
            branches.append(f"({branch})")
        return "WHERE " + " OR ".join(branches), params

    @staticmethod
    def _encode_cursor(payload: Mapping[str, Any]) -> str:
        return base64.urlsafe_b64encode(_json(payload).encode()).decode().rstrip("=")

    @staticmethod
    def _decode_cursor(cursor: str) -> dict[str, Any]:
        try:
            padding = "=" * (-len(cursor) % 4)
            value = json.loads(base64.urlsafe_b64decode(cursor + padding))
        except (ValueError, json.JSONDecodeError) as error:
            raise NotionToolError("invalid_cursor", "Cursor is not valid") from error
        if not isinstance(value, dict):
            raise NotionToolError("invalid_cursor", "Cursor is not valid")
        return value


def _condition_field(kind: str) -> str:
    if kind == "number":
        return "p.value_number"
    if kind == "checkbox":
        return "p.value_boolean"
    if kind == "date":
        return "p.value_time"
    if kind in {
        "title",
        "rich_text",
        "select",
        "status",
        "multi_select",
        "people",
        "relation",
        "url",
        "email",
        "phone_number",
    }:
        return "p.value_text"
    raise NotionToolError("invalid_filter", f"Unsupported property type: {kind}")


def _property_condition(
    field: str, operator: str, value: Any, *, direct: bool = False
) -> tuple[str, list[Any], bool]:
    column = field if direct else field
    if operator == "equals":
        return (
            f"{column} = ?",
            [int(value) if isinstance(value, bool) else value],
            False,
        )
    if operator == "does_not_equal":
        return f"{column} = ?", [int(value) if isinstance(value, bool) else value], True
    if operator in {"greater_than", "after"}:
        return f"{column} > ?", [value], False
    if operator in {"less_than", "before"}:
        return f"{column} < ?", [value], False
    if operator in {"greater_than_or_equal_to", "on_or_after"}:
        return f"{column} >= ?", [value], False
    if operator in {"less_than_or_equal_to", "on_or_before"}:
        return f"{column} <= ?", [value], False
    escaped = str(value).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    if operator == "contains":
        return f"{column} LIKE ? ESCAPE '\\'", [f"%{escaped}%"], False
    if operator == "does_not_contain":
        return f"{column} LIKE ? ESCAPE '\\'", [f"%{escaped}%"], True
    if operator == "starts_with":
        return f"{column} LIKE ? ESCAPE '\\'", [f"{escaped}%"], False
    if operator == "ends_with":
        return f"{column} LIKE ? ESCAPE '\\'", [f"%{escaped}"], False
    raise NotionToolError("invalid_filter", f"Unsupported filter operator: {operator}")


def _normalize_id(value: str) -> str:
    value = value.strip()
    if value.startswith("notion://page/"):
        value = value.rsplit("/", 1)[-1]
    try:
        return str(uuid.UUID(value))
    except ValueError as error:
        raise NotionToolError("invalid_id", f"Invalid Notion ID: {value}") from error


_DEFAULT_TOOLS: NotionTools | None = None
_DEFAULT_LOCK = threading.Lock()


def _default_tools() -> NotionTools:
    global _DEFAULT_TOOLS
    if _DEFAULT_TOOLS is None:
        with _DEFAULT_LOCK:
            if _DEFAULT_TOOLS is None:
                _DEFAULT_TOOLS = NotionTools()
    return _DEFAULT_TOOLS
