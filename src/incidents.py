"""Small JSONL incident store for hackathon evidence packages."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any


class IncidentStore:
    """Persist incident evidence without introducing a database dependency."""

    def __init__(self, path: str | Path = "data/incidents.jsonl") -> None:
        self.path = Path(path)
        self._lock = Lock()

    def append(self, incident: dict[str, Any]) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(incident, sort_keys=True) + "\n")
        return incident

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        rows: list[dict[str, Any]] = []
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(rows) >= limit:
                break
        return rows

    def get(self, incident_id: str) -> dict[str, Any] | None:
        for incident in self.list(limit=1000):
            if incident.get("incident_id") == incident_id:
                return incident
        return None

    def update(self, incident_id: str, fields: dict[str, Any]) -> dict[str, Any] | None:
        if not self.path.exists():
            return None

        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
            rows: list[dict[str, Any]] = []
            target_index: int | None = None
            for line in lines:
                if not line.strip():
                    continue
                try:
                    incident = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if incident.get("incident_id") == incident_id:
                    target_index = len(rows)
                rows.append(incident)

            if target_index is None:
                return None

            rows[target_index].update(fields)
            with self.path.open("w", encoding="utf-8") as handle:
                for incident in rows:
                    handle.write(json.dumps(incident, sort_keys=True) + "\n")
            return rows[target_index]
