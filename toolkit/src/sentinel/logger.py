"""Append-only JSONL log store.

The logger writes one JSON object per line to a file, creating a canonical,
streaming-friendly experiment log that can be consumed by
offline analysis scripts or the reporting UI.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from sentinel.models import EventType, LogEvent, ModelResponse, PromptCandidate, Verdict


class JsonlLogger:
    """Append-only JSONL log writer.

    Parameters
    ----------
    path : str | Path
        Destination file.  Parent directories are created automatically.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a", encoding="utf-8")

    # -- low-level -----------------------------------------------------------

    def _write(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        self._file.write(line + "\n")
        self._file.flush()

    # -- high-level helpers --------------------------------------------------

    def log_event(self, event: LogEvent) -> None:
        """Write an arbitrary `LogEvent` to the log file."""
        self._write(event.to_dict())

    def log_experiment_start(
        self,
        experiment_id: str,
        manifest: Dict[str, Any],
    ) -> None:
        self._write(
            LogEvent(
                event_type=EventType.EXPERIMENT_START.value,
                experiment_id=experiment_id,
                data={"manifest": manifest},
            ).to_dict()
        )

    def log_experiment_end(
        self,
        experiment_id: str,
        summary: Dict[str, Any] | None = None,
    ) -> None:
        self._write(
            LogEvent(
                event_type=EventType.EXPERIMENT_END.value,
                experiment_id=experiment_id,
                data={"summary": summary or {}},
            ).to_dict()
        )

    def log_trial(
        self,
        experiment_id: str,
        prompt: PromptCandidate,
        response: ModelResponse,
        verdicts: list[Verdict],
    ) -> None:
        """Log a complete trial: prompt -> response -> verdicts."""
        self._write(
            LogEvent(
                event_type=EventType.TRIAL_RESULT.value,
                experiment_id=experiment_id,
                data={
                    "prompt": prompt.to_dict(),
                    "response": response.to_dict(),
                    "verdicts": [v.to_dict() for v in verdicts],
                },
            ).to_dict()
        )

    def log_error(
        self,
        experiment_id: str,
        message: str,
        details: Dict[str, Any] | None = None,
    ) -> None:
        self._write(
            LogEvent(
                event_type=EventType.ERROR.value,
                experiment_id=experiment_id,
                data={"message": message, **(details or {})},
            ).to_dict()
        )

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
