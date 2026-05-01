"""Core data models shared across all toolkit modules.

Every struct uses frozen dataclasses / Pydantic-style models so that they
are hashable, serialisable and easy to log in JSONL format.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class InferenceMode(str, Enum):
    """Declares what kind of model access an adapter provides."""
    BLACK_BOX = "black_box"
    WHITE_BOX = "white_box"


class JudgeType(str, Enum):
    """Judge implementation category - used in verdicts and manifests."""
    HEURISTIC = "heuristic"
    CLASSIFIER = "classifier"
    LLM = "llm"
    ENSEMBLE = "ensemble"


class HealthStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class EventType(str, Enum):
    """Types of events recorded in the JSONL log store."""
    EXPERIMENT_START = "experiment_start"
    EXPERIMENT_END = "experiment_end"
    TRIAL_RESULT = "trial_result"
    ERROR = "error"
    INFO = "info"


# ---------------------------------------------------------------------------
# Data-transfer objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptCandidate:
    """A single adversarial prompt produced by an attack generator."""
    prompt_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CostInfo:
    """Estimated cost / resource usage for a single inference call."""
    estimated_tokens: int = 0
    estimated_cost_usd: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    """Normalised response returned by a model adapter."""
    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt_id: str = ""
    model_id: str = ""
    adapter_instance_id: str = ""
    text: str = ""
    tokens: int = 0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    # White-box fields — populated when inference_mode is white_box.
    logprobs: List[Dict[str, Any]] = field(default_factory=list)
    perplexity: Optional[float] = None
    token_entropies: List[float] = field(default_factory=list)
    mean_entropy: Optional[float] = None
    top1_probs: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Verdict:
    """Structured evaluation verdict produced by a judge."""
    verdict_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experiment_id: str = ""
    prompt_id: str = ""
    model_id: str = ""
    judge_instance_id: str = ""
    trace_id: str = ""
    labels: List[str] = field(default_factory=list)
    confidence: float = 0.0
    judge_type: str = JudgeType.HEURISTIC.value
    explanation: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LogEvent:
    """A single JSONL log record."""
    event_type: str = EventType.INFO.value
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    experiment_id: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
