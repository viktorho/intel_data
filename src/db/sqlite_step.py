from __future__ import annotations
from typing import Any, AsyncIterable, Callable, Dict, Optional, Tuple, List
import asyncio, json, aiosqlite

INIT_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  topic TEXT NOT NULL,
  payload TEXT NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_events_topic_id ON events(topic, id);

CREATE TABLE IF NOT EXISTS consumer_offsets (
  consumer TEXT PRIMARY KEY,
  last_id  INTEGER NOT NULL
);
"""

class SQLiteStepStore:
    """
    A small, robust SQLite helper for step-level streaming.
    - Append events (JSON) while your step runs
    - Pipe from an async generator (listen & write in sync)
    - Stream for consumers and checkpoint offsets
    """
    def __init__(
        self,
        db_path: str = "events.db",
        topic: str = "default.q",
        *,
        batch_size: int = 100,
    ):
        self.db_path = db_path
        self.topic = topic
        self.batch_size = batch_size

        self._db: Optional[aiosqlite.Connection] = None
        self._batch: List[Tuple[str, str]] = []
        self._lock = asyncio.Lock()  # serialize writes
        self._inserted = 0
        self._last_event_id: Optional[int] = None

    # ---------- lifecycle ----------
    async def open(self) -> "SQLiteStepStore":
        self._db = await aiosqlite.connect(self.db_path)
        # Apply all init statements
        for stmt in filter(None, (s.strip() for s in INIT_SQL.split(";"))):
            await self._db.execute(stmt)
        await self._db.commit()
        return self

    async def close(self) -> None:
        if not self._db:
            return
        await self.flush()
        await self._db.close()
        self._db = None

    async def __aenter__(self) -> "SQLiteStepStore":
        return await self.open()

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    # ---------- write path ----------
    async def append(self, item: Any) -> None:
        """
        Queue a single item (dict/str/whatever JSON-serializable).
        Flushes when batch_size is reached.
        """
        assert self._db is not None, "call open() first"
        payload = json.dumps(item, ensure_ascii=False)
        async with self._lock:
            self._batch.append((self.topic, payload))
            if len(self._batch) >= self.batch_size:
                await self._flush_locked()

    async def append_many(self, items: List[Any]) -> None:
        assert self._db is not None, "call open() first"
        rows = [(self.topic, json.dumps(it, ensure_ascii=False)) for it in items]
        async with self._lock:
            self._batch.extend(rows)
            if len(self._batch) >= self.batch_size:
                await self._flush_locked()

    async def flush(self) -> None:
        assert self._db is not None, "call open() first"
        async with self._lock:
            await self._flush_locked()

    async def _flush_locked(self) -> None:
        """Flush current batch. Caller must hold _lock."""
        if not self._batch:
            return
        cur = await self._db.executemany(
            "INSERT INTO events(topic, payload) VALUES (?, ?)",
            self._batch
        )
        await self._db.commit()
        # last_insert_rowid() is per-connection; safe here
        async with self._db.execute("SELECT last_insert_rowid()") as c2:
            row = await c2.fetchone()
        self._last_event_id = int(row[0]) if row else self._last_event_id
        self._inserted += len(self._batch)
        self._batch.clear()

    # ---------- pipe from async generator ----------
    async def pipe_from(
        self,
        agen: AsyncIterable[Any],
        *,
        transform: Optional[Callable[[Any], Any]] = None,
        on_chunk: Optional[Callable[[Any], None]] = None,
    ) -> Dict[str, Any]:
        """
        Consume an async iterable; write each chunk as an event.
        - transform: optional mapper per chunk before JSON
        - on_chunk: optional side-effect sink (e.g., UI logger)
        Returns a minimal summary for successor nodes.
        """
        async for chunk in agen:
            if transform:
                chunk = transform(chunk)
            if on_chunk:
                on_chunk(chunk)
            await self.append(chunk)
        await self.flush()
        return self.summary()

    # ---------- read path for successors ----------
    async def ensure_consumer(self, consumer: str) -> None:
        assert self._db is not None, "call open() first"
        await self._db.execute(
            "INSERT OR IGNORE INTO consumer_offsets(consumer, last_id) VALUES(?, 0)",
            (consumer,)
        )
        await self._db.commit()

    async def get_offset(self, consumer: str) -> int:
        assert self._db is not None, "call open() first"
        async with self._db.execute(
            "SELECT last_id FROM consumer_offsets WHERE consumer=?",
            (consumer,)
        ) as cur:
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def set_offset(self, consumer: str, last_id: int) -> None:
        assert self._db is not None, "call open() first"
        await self._db.execute(
            "UPDATE consumer_offsets SET last_id=? WHERE consumer=?",
            (last_id, consumer)
        )
        await self._db.commit()

    async def stream(
        self,
        *,
        since_id: int = 0,
        page: int = 256,
        poll_ms: int = 150
    ):
        """
        Async generator for reading new events for self.topic.
        Yields (id, payload_dict).
        """
        assert self._db is not None, "call open() first"
        while True:
            async with self._db.execute(
                "SELECT id, payload FROM events "
                "WHERE topic=? AND id>? ORDER BY id ASC LIMIT ?",
                (self.topic, since_id, page)
            ) as cur:
                rows = await cur.fetchall()

            if not rows:
                await asyncio.sleep(poll_ms / 1000)
                continue

            for eid, payload in rows:
                since_id = int(eid)
                yield since_id, json.loads(payload)

    # ---------- summary ----------
    def summary(self) -> Dict[str, Any]:
        return {
            "db_path": self.db_path,
            "topic": self.topic,
            "inserted": self._inserted,
            "last_event_id": self._last_event_id,
        }
