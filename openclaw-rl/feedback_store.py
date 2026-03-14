"""Thread-safe feedback data store for reward model training.

Collects user thumbs-down (👎) signals and stores (prompt, response, label)
tuples in a JSONL file.  Provides batched sampling for reward-model training.
"""

import json
import logging
import os
import random
import threading
import time
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)


@dataclass
class FeedbackRecord:
    """A single feedback record."""

    session_id: str
    turn: int
    prompt_text: str
    response_text: str
    label: int  # -1 = thumbs-down,  +1 = implicit positive
    timestamp: str = ""
    message_index: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "FeedbackRecord":
        return FeedbackRecord(**{k: v for k, v in d.items() if k in FeedbackRecord.__dataclass_fields__})


class FeedbackStore:
    """Thread-safe persistent store for user feedback records.

    Records are appended to a JSONL file and kept in-memory for fast training
    batch sampling.  The store distinguishes between *explicit negatives*
    (user clicked 👎) and *implicit positives* (turns that were submitted to
    training without a negative signal).
    """

    def __init__(
        self,
        store_path: str = "",
        max_records: int = 100_000,
    ):
        self._lock = threading.Lock()
        self._records: list[FeedbackRecord] = []
        self._max_records = max_records
        self._store_path = store_path

        if self._store_path:
            os.makedirs(os.path.dirname(self._store_path) or ".", exist_ok=True)
            self._load_from_disk()

    # ------------------------------------------------------------------ I/O

    def _load_from_disk(self):
        """Load existing records from the JSONL file."""
        if not self._store_path or not os.path.exists(self._store_path):
            return
        loaded = 0
        try:
            with open(self._store_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = FeedbackRecord.from_dict(json.loads(line))
                        self._records.append(rec)
                        loaded += 1
                    except Exception:
                        continue
            logger.info("[FeedbackStore] loaded %d records from %s", loaded, self._store_path)
        except OSError as e:
            logger.warning("[FeedbackStore] failed to load store: %s", e)

    def _append_to_disk(self, record: FeedbackRecord):
        if not self._store_path:
            return
        try:
            with open(self._store_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("[FeedbackStore] failed to write record: %s", e)

    def _rewrite_disk(self):
        """Rewrite the entire store file (used after eviction)."""
        if not self._store_path:
            return
        try:
            with open(self._store_path, "w", encoding="utf-8") as f:
                for rec in self._records:
                    f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("[FeedbackStore] failed to rewrite store: %s", e)

    # ------------------------------------------------------------------ API

    def add_negative(
        self,
        session_id: str,
        turn: int,
        prompt_text: str,
        response_text: str,
        message_index: int | None = None,
    ):
        """Record a user thumbs-down signal."""
        rec = FeedbackRecord(
            session_id=session_id,
            turn=turn,
            prompt_text=prompt_text,
            response_text=response_text,
            label=-1,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            message_index=message_index,
        )
        with self._lock:
            replaced = self._replace_record(session_id, turn, rec)
            if replaced:
                self._rewrite_disk()
            else:
                self._append_to_disk(rec)
            self._maybe_evict()
            total = len(self._records)
        logger.info(
            "[FeedbackStore] added NEGATIVE session=%s turn=%d (total=%d)",
            session_id,
            turn,
            total,
        )

    def add_positive(
        self,
        session_id: str,
        turn: int,
        prompt_text: str,
        response_text: str,
    ):
        """Record an implicit positive (turn submitted without thumbs-down)."""
        rec = FeedbackRecord(
            session_id=session_id,
            turn=turn,
            prompt_text=prompt_text,
            response_text=response_text,
            label=1,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        with self._lock:
            if any(r.session_id == session_id and r.turn == turn for r in self._records):
                return
            self._records.append(rec)
            self._append_to_disk(rec)
            self._maybe_evict()

    def get_latest_record(self, session_id: str, turn: int | None = None) -> FeedbackRecord | None:
        with self._lock:
            for rec in reversed(self._records):
                if rec.session_id != session_id:
                    continue
                if turn is not None and rec.turn != turn:
                    continue
                return rec
        return None

    def _replace_record(self, session_id: str, turn: int, new_record: FeedbackRecord) -> bool:
        replaced = False
        kept_records = []
        for rec in self._records:
            if rec.session_id == session_id and rec.turn == turn:
                replaced = True
                continue
            kept_records.append(rec)
        kept_records.append(new_record)
        self._records = kept_records
        return replaced

    def _maybe_evict(self):
        """Evict oldest records if over capacity (caller must hold lock)."""
        if len(self._records) > self._max_records:
            excess = len(self._records) - self._max_records
            self._records = self._records[excess:]
            self._rewrite_disk()
            logger.info("[FeedbackStore] evicted %d old records", excess)

    # ---------------------------------------------------- training helpers

    def sample_batch(
        self,
        batch_size: int,
        balance: bool = True,
    ) -> list[FeedbackRecord]:
        """Sample a balanced batch for reward model training.

        If *balance* is True, we sample equally from positives and negatives
        (up to available counts), which is important because negatives are
        typically much rarer.
        """
        with self._lock:
            if not self._records:
                return []

            if not balance:
                return random.sample(self._records, min(batch_size, len(self._records)))

            positives = [r for r in self._records if r.label == 1]
            negatives = [r for r in self._records if r.label == -1]

            if not negatives:
                return random.sample(positives, min(batch_size, len(positives)))
            if not positives:
                return random.sample(negatives, min(batch_size, len(negatives)))

            half = batch_size // 2
            n_neg = min(half, len(negatives))
            n_pos = min(batch_size - n_neg, len(positives))

            batch = random.sample(negatives, n_neg) + random.sample(positives, n_pos)
            random.shuffle(batch)
            return batch

    def get_all_records(self) -> list[FeedbackRecord]:
        with self._lock:
            return list(self._records)

    @property
    def num_positives(self) -> int:
        with self._lock:
            return sum(1 for r in self._records if r.label == 1)

    @property
    def num_negatives(self) -> int:
        with self._lock:
            return sum(1 for r in self._records if r.label == -1)

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def stats(self) -> dict:
        with self._lock:
            n_pos = sum(1 for r in self._records if r.label == 1)
            n_neg = sum(1 for r in self._records if r.label == -1)
            return {"total": len(self._records), "positives": n_pos, "negatives": n_neg}
