"""Itemized, attributable review knowledge.

Learned guidance is never appended to a prompt blob. Every lesson is a
``KnowledgeItem`` with its own triggers, evidence, lifecycle status and
outcome counters, so it can be retrieved selectively, credited or blamed for
the findings it influenced, merged with near-duplicates and evicted when it
stops paying for its tokens.
"""
import json
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional

from .context_manager import estimate_tokens
from .similarity import fingerprint as text_fingerprint


KINDS = ("rule", "skill")
STATUSES = ("probation", "active", "archived")
ROLES = ("*", "lead", "security", "correctness-reliability", "critic")
DEFAULT_HALF_LIFE_DAYS = 30.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_time(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def days_between(earlier: str, later: str) -> float:
    start, end = _parse_time(earlier), _parse_time(later)
    if start is None or end is None:
        return 0.0
    return max(0.0, (end - start).total_seconds() / 86400.0)


def decay_factor(elapsed_days: float, half_life_days: float) -> float:
    if half_life_days <= 0:
        return 1.0
    return 0.5 ** (max(0.0, elapsed_days) / half_life_days)


def new_item_id() -> str:
    return "K-" + uuid.uuid4().hex[:8]


@dataclass
class KnowledgeItem:
    tenant_id: str
    kind: str
    content: str
    role: str = "*"
    scope: str = "*"
    name: str = ""
    body: str = ""
    triggers: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "probation"
    pinned: bool = False
    source: str = "manual"
    id: str = field(default_factory=new_item_id)
    fingerprint: str = ""
    tokens: int = 0
    helpful: float = 0.0
    harmful: float = 0.0
    helpful_total: float = 0.0
    harmful_total: float = 0.0
    exposures: int = 0
    citations: int = 0
    decayed_at: str = ""
    last_used_at: str = ""
    last_cited_at: str = ""
    version: int = 1
    parent_ids: List[str] = field(default_factory=list)
    archived_reason: str = ""
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError("knowledge kind must be one of %s" % ", ".join(KINDS))
        if self.status not in STATUSES:
            raise ValueError("knowledge status must be one of %s" % ", ".join(STATUSES))
        if self.role not in ROLES:
            raise ValueError("knowledge role must be one of %s" % ", ".join(ROLES))
        self.content = str(self.content or "").strip()
        self.body = str(self.body or "").strip()
        if not self.content:
            raise ValueError("knowledge content must not be empty")
        self.refresh_derived()
        now = utc_now()
        self.created_at = self.created_at or now
        self.updated_at = self.updated_at or now
        self.decayed_at = self.decayed_at or self.created_at

    def refresh_derived(self) -> None:
        self.fingerprint = text_fingerprint(self.content + " " + self.body)
        self.tokens = estimate_tokens(self.instruction_text())
        self.triggers = list(dict.fromkeys(
            str(item).strip().lower() for item in self.triggers if str(item).strip()
        ))[:24]

    def instruction_text(self) -> str:
        """The exact text a role pays for when this item is injected."""
        if self.kind == "skill":
            return "%s: %s\n%s" % (self.name, self.content, self.body)
        return self.content

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "KnowledgeItem":
        known = {name for name in cls.__dataclass_fields__}
        data = {key: value[key] for key in value if key in known}
        return cls(**data)


def utility(item: KnowledgeItem, now: str = "", half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> float:
    """Expected value of injecting an item, in [0, 1].

    A Beta(1, 1)-smoothed helpful/harmful ratio over time-decayed counters,
    scaled by how often the item is actually cited when it is shown. A new
    item scores 0.375; an item shown 40 times and never cited sinks below
    0.27 even without harmful outcomes, which is the "useless" rot the
    eviction policy targets.
    """
    factor = decay_factor(days_between(item.decayed_at, now or utc_now()), half_life_days)
    helpful, harmful = item.helpful * factor, item.harmful * factor
    precision = (helpful + 1.0) / (helpful + harmful + 2.0)
    citation_rate = (item.citations + 1.0) / (item.exposures + 2.0)
    return round(precision * (0.5 + 0.5 * citation_rate), 6)


_COLUMNS = (
    "id", "tenant_id", "kind", "role", "scope", "name", "content", "body",
    "triggers_json", "evidence_json", "status", "pinned", "source", "fingerprint",
    "tokens", "helpful", "harmful", "helpful_total", "harmful_total", "exposures",
    "citations", "decayed_at", "last_used_at", "last_cited_at", "version",
    "parent_ids_json", "archived_reason", "created_at", "updated_at",
)
_JSON_FIELDS = {"triggers": "triggers_json", "evidence": "evidence_json", "parent_ids": "parent_ids_json"}


class KnowledgeRepository:
    """Portable SQL storage for knowledge items, audit events and releases.

    Works over any DB-API connection factory whose connections are context
    managers that commit and close (``TaskStore._connect`` and psycopg).
    """

    def __init__(
        self, connect: Callable[[], Any], placeholder: str = "?",
        lock: Optional[threading.Lock] = None, row_locks: bool = False,
    ):
        self._connect = connect
        self._p = placeholder
        self._lock = lock or threading.Lock()
        self._row_locks = row_locks
        self.init_schema()

    @contextmanager
    def _write(self):
        with self._lock, self._connect() as conn:
            yield conn

    def _sql(self, statement: str) -> str:
        return statement.replace("?", self._p)

    def init_schema(self) -> None:
        with self._write() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS knowledge_items (
                    id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, kind TEXT NOT NULL,
                    role TEXT NOT NULL, scope TEXT NOT NULL, name TEXT NOT NULL,
                    content TEXT NOT NULL, body TEXT NOT NULL, triggers_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL, status TEXT NOT NULL, pinned INTEGER NOT NULL,
                    source TEXT NOT NULL, fingerprint TEXT NOT NULL, tokens INTEGER NOT NULL,
                    helpful DOUBLE PRECISION NOT NULL, harmful DOUBLE PRECISION NOT NULL,
                    helpful_total DOUBLE PRECISION NOT NULL,
                    harmful_total DOUBLE PRECISION NOT NULL, exposures INTEGER NOT NULL,
                    citations INTEGER NOT NULL, decayed_at TEXT NOT NULL,
                    last_used_at TEXT NOT NULL, last_cited_at TEXT NOT NULL,
                    version INTEGER NOT NULL, parent_ids_json TEXT NOT NULL,
                    archived_reason TEXT NOT NULL, created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_items_tenant_status "
                "ON knowledge_items(tenant_id, status, kind)"
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS knowledge_events (
                    id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, item_id TEXT NOT NULL,
                    event TEXT NOT NULL, detail_json TEXT NOT NULL, created_at TEXT NOT NULL
                )"""
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_events_tenant "
                "ON knowledge_events(tenant_id, created_at)"
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS knowledge_releases (
                    id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, version INTEGER NOT NULL,
                    status TEXT NOT NULL, reason TEXT NOT NULL, snapshot_json TEXT NOT NULL,
                    gate_json TEXT NOT NULL, created_at TEXT NOT NULL
                )"""
            )

    # ----------------------------------------------------------------- items
    @staticmethod
    def _row(item: KnowledgeItem) -> List[Any]:
        data = item.to_dict()
        values = []
        for column in _COLUMNS:
            field_name = next(
                (key for key, value in _JSON_FIELDS.items() if value == column), column
            )
            value = data[field_name]
            if column in _JSON_FIELDS.values():
                value = json.dumps(value, ensure_ascii=False, sort_keys=True)
            elif column == "pinned":
                value = int(bool(value))
            values.append(value)
        return values

    @staticmethod
    def _decode(row: Any) -> KnowledgeItem:
        data = dict(row)
        for field_name, column in _JSON_FIELDS.items():
            data[field_name] = json.loads(data.pop(column) or "[]")
        data["pinned"] = bool(data.get("pinned"))
        return KnowledgeItem.from_dict(data)

    def create(self, item: KnowledgeItem) -> KnowledgeItem:
        item.refresh_derived()
        with self._write() as conn:
            conn.execute(
                self._sql("INSERT INTO knowledge_items(%s) VALUES (%s)" % (
                    ",".join(_COLUMNS), ",".join("?" for _ in _COLUMNS)
                )), self._row(item),
            )
        return item

    def save(self, item: KnowledgeItem) -> KnowledgeItem:
        """Overwrite an existing item (content edits must bump ``version``)."""
        item.refresh_derived()
        item.updated_at = utc_now()
        columns = [column for column in _COLUMNS if column not in {"id", "tenant_id"}]
        values = self._row(item)
        by_column = dict(zip(_COLUMNS, values))
        with self._write() as conn:
            conn.execute(
                self._sql("UPDATE knowledge_items SET %s WHERE id=? AND tenant_id=?" % ",".join(
                    "%s=?" % column for column in columns
                )), [by_column[column] for column in columns] + [item.id, item.tenant_id],
            )
        return item

    def get(self, tenant_id: str, item_id: str) -> Optional[KnowledgeItem]:
        with self._connect() as conn:
            row = conn.execute(
                self._sql("SELECT * FROM knowledge_items WHERE tenant_id=? AND id=?"),
                (tenant_id, item_id),
            ).fetchone()
        return self._decode(row) if row else None

    def list(
        self, tenant_id: str, statuses: Iterable[str] = (), kind: str = "",
        limit: int = 5000,
    ) -> List[KnowledgeItem]:
        clauses, params = ["tenant_id=?"], [tenant_id]
        statuses = [item for item in statuses if item]
        if statuses:
            clauses.append("status IN (%s)" % ",".join("?" for _ in statuses))
            params.extend(statuses)
        if kind:
            clauses.append("kind=?")
            params.append(kind)
        with self._connect() as conn:
            rows = conn.execute(
                self._sql(
                    "SELECT * FROM knowledge_items WHERE %s ORDER BY created_at, id LIMIT ?"
                    % " AND ".join(clauses)
                ), params + [int(limit)],
            ).fetchall()
        return [self._decode(row) for row in rows]

    def find_fingerprint(
        self, tenant_id: str, value: str, kind: str,
    ) -> Optional[KnowledgeItem]:
        with self._connect() as conn:
            row = conn.execute(
                self._sql(
                    "SELECT * FROM knowledge_items WHERE tenant_id=? AND fingerprint=? "
                    "AND kind=? AND status<>'archived' ORDER BY created_at LIMIT 1"
                ), (tenant_id, value, kind),
            ).fetchone()
        return self._decode(row) if row else None

    def apply_outcome(
        self, tenant_id: str, item_id: str, helpful: float = 0.0, harmful: float = 0.0,
        exposures: int = 0, citations: int = 0, at: str = "",
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        event: str = "", detail: Optional[Dict[str, Any]] = None,
    ) -> Optional[KnowledgeItem]:
        """Decay then increment an item's counters atomically."""
        at = at or utc_now()
        select = "SELECT * FROM knowledge_items WHERE tenant_id=? AND id=?"
        if self._row_locks:
            select += " FOR UPDATE"
        with self._write() as conn:
            row = conn.execute(self._sql(select), (tenant_id, item_id)).fetchone()
            if not row:
                return None
            item = self._decode(row)
            if event and detail and detail.get("idempotency_key"):
                previous = conn.execute(self._sql(
                    "SELECT detail_json FROM knowledge_events WHERE tenant_id=? AND item_id=? AND event=?"
                ), (tenant_id, item_id, event)).fetchall()
                if any(json.loads(dict(record)["detail_json"]).get("idempotency_key") == detail["idempotency_key"]
                       for record in previous):
                    return item
            factor = decay_factor(days_between(item.decayed_at, at), half_life_days)
            item.helpful = round(item.helpful * factor + max(0.0, helpful), 6)
            item.harmful = round(item.harmful * factor + max(0.0, harmful), 6)
            item.helpful_total = round(item.helpful_total + max(0.0, helpful), 6)
            item.harmful_total = round(item.harmful_total + max(0.0, harmful), 6)
            item.exposures += max(0, int(exposures))
            item.citations += max(0, int(citations))
            item.decayed_at = at
            if exposures or citations:
                item.last_used_at = at
            if citations:
                item.last_cited_at = at
            conn.execute(
                self._sql(
                    "UPDATE knowledge_items SET helpful=?, harmful=?, helpful_total=?, "
                    "harmful_total=?, exposures=?, citations=?, decayed_at=?, "
                    "last_used_at=?, last_cited_at=? WHERE tenant_id=? AND id=?"
                ), (
                    item.helpful, item.harmful, item.helpful_total, item.harmful_total,
                    item.exposures, item.citations, item.decayed_at, item.last_used_at,
                    item.last_cited_at, tenant_id, item_id,
                ),
            )
            if event:
                conn.execute(self._sql(
                    "INSERT INTO knowledge_events(id,tenant_id,item_id,event,detail_json,created_at) "
                    "VALUES (?,?,?,?,?,?)"
                ), (uuid.uuid4().hex, tenant_id, item_id, event,
                    json.dumps(detail or {}, ensure_ascii=False, default=str), at))
        return item

    def set_status(
        self, tenant_id: str, item_id: str, status: str, reason: str = "",
    ) -> Optional[KnowledgeItem]:
        if status not in STATUSES:
            raise ValueError("unsupported knowledge status")
        item = self.get(tenant_id, item_id)
        if item is None:
            return None
        item.status = status
        item.archived_reason = reason if status == "archived" else ""
        return self.save(item)

    def stats(self, tenant_id: str) -> Dict[str, Any]:
        items = self.list(tenant_id)
        summary: Dict[str, Any] = {
            "items": len(items),
            "by_status": {status: 0 for status in STATUSES},
            "by_kind": {kind: 0 for kind in KINDS},
            "live_tokens": 0,
        }
        for item in items:
            summary["by_status"][item.status] += 1
            summary["by_kind"][item.kind] += 1
            if item.status != "archived":
                summary["live_tokens"] += item.tokens
        return summary

    # ---------------------------------------------------------------- events
    def log_event(
        self, tenant_id: str, item_id: str, event: str, detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._write() as conn:
            conn.execute(
                self._sql(
                    "INSERT INTO knowledge_events(id,tenant_id,item_id,event,detail_json,created_at) "
                    "VALUES (?,?,?,?,?,?)"
                ), (
                    uuid.uuid4().hex, tenant_id, item_id or "", event,
                    json.dumps(detail or {}, ensure_ascii=False, default=str), utc_now(),
                ),
            )

    def events(
        self, tenant_id: str, item_id: str = "", limit: int = 200,
    ) -> List[Dict[str, Any]]:
        clauses, params = ["tenant_id=?"], [tenant_id]
        if item_id:
            clauses.append("item_id=?")
            params.append(item_id)
        with self._connect() as conn:
            rows = conn.execute(
                self._sql(
                    "SELECT * FROM knowledge_events WHERE %s ORDER BY created_at DESC, id LIMIT ?"
                    % " AND ".join(clauses)
                ), params + [int(limit)],
            ).fetchall()
        values = []
        for row in rows:
            value = dict(row)
            value["detail"] = json.loads(value.pop("detail_json") or "{}")
            values.append(value)
        return values

    # -------------------------------------------------------------- releases
    def save_release(
        self, tenant_id: str, status: str, reason: str, gate: Dict[str, Any],
        items: Optional[List[KnowledgeItem]] = None,
    ) -> Dict[str, Any]:
        snapshot = [item.to_dict() for item in (
            items if items is not None
            else self.list(tenant_id, statuses=("active", "probation"))
        )]
        with self._write() as conn:
            row = conn.execute(
                self._sql(
                    "SELECT COALESCE(MAX(version),0) AS version FROM knowledge_releases "
                    "WHERE tenant_id=?"
                ), (tenant_id,),
            ).fetchone()
            version = int(dict(row)["version"]) + 1
            release = {
                "id": uuid.uuid4().hex, "tenant_id": tenant_id, "version": version,
                "status": status, "reason": reason, "created_at": utc_now(),
            }
            conn.execute(
                self._sql(
                    "INSERT INTO knowledge_releases(id,tenant_id,version,status,reason,"
                    "snapshot_json,gate_json,created_at) VALUES (?,?,?,?,?,?,?,?)"
                ), (
                    release["id"], tenant_id, version, status, reason,
                    json.dumps(snapshot, ensure_ascii=False, default=str),
                    json.dumps(gate, ensure_ascii=False, default=str),
                    release["created_at"],
                ),
            )
        return {**release, "items": len(snapshot), "gate": gate}

    def list_releases(self, tenant_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                self._sql(
                    "SELECT id,tenant_id,version,status,reason,gate_json,created_at "
                    "FROM knowledge_releases WHERE tenant_id=? ORDER BY version DESC LIMIT ?"
                ), (tenant_id, int(limit)),
            ).fetchall()
        values = []
        for row in rows:
            value = dict(row)
            value["gate"] = json.loads(value.pop("gate_json") or "{}")
            values.append(value)
        return values

    def release_snapshot(self, tenant_id: str, version: int) -> Optional[List[KnowledgeItem]]:
        with self._connect() as conn:
            row = conn.execute(
                self._sql(
                    "SELECT snapshot_json FROM knowledge_releases WHERE tenant_id=? AND version=?"
                ), (tenant_id, int(version)),
            ).fetchone()
        if not row:
            return None
        return [KnowledgeItem.from_dict(item) for item in json.loads(dict(row)["snapshot_json"])]
