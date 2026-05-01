"""Local llama.cpp-backed model adapter.

Supports two inference modes:

* ``black_box`` (default) — only prompt/response text is returned.

* ``white_box`` — additionally populates ``ModelResponse`` with per-token
  log-probabilities, perplexity, per-token entropy, mean entropy, and
  top-1 probabilities.  Also enables direct logit surgery via
  ``logit_bias_tokens`` and LoRA adapter injection via ``lora_path``.
"""

from __future__ import annotations

import asyncio
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from llama_cpp import Llama

from sentinel.model_adapters.base import ModelAdapter
from sentinel.models import (
    CostInfo,
    HealthStatus,
    InferenceMode,
    ModelResponse,
)


class LlamaCppModelAdapter(ModelAdapter):
    """Serve model responses from a local GGUF model via llama.cpp.

    Black-box configuration keys
    ----------------------------
    model_path : str
        Path to GGUF model file (required).
    inference_mode : str
        ``"black_box"`` (default) or ``"white_box"``.
    system_prompt : str
        System message prepended to every request.
    max_tokens : int
        Maximum generated tokens per response (default 256).
    temperature : float
        Sampling temperature (default 0.7).
    n_ctx : int
        Context window size in tokens (default 2048).
    n_batch : int
        Logical batch size for prompt processing (default 512).
    n_gpu_layers : int
        Layers to offload to GPU (default 0, CPU only).
    n_threads : int
        CPU inference threads (default: auto).
    n_threads_batch : int
        CPU threads for batch/prompt processing (default: auto).
    flash_attn : bool
        Enable flash attention if the backend supports it (default False).
    offload_kqv : bool
        Offload K/V cache to GPU when n_gpu_layers > 0 (default True).
    verbose : bool
        Enable llama.cpp verbose logs (default False).

    Sampling parameters (all modes)
    --------------------------------
    top_p : float
        Nucleus sampling cut-off (default 0.95).
    top_k : int
        Top-K sampling limit (default 40).
    min_p : float
        Minimum probability threshold (default 0.05).
    typical_p : float
        Locally-typical sampling (default 1.0 = disabled).
    tfs_z : float
        Tail-free sampling z parameter (default 1.0 = disabled).
    repeat_penalty : float
        Repetition penalty (default 1.1; 1.0 = disabled).
    presence_penalty : float
        Presence penalty — penalises any repeated token (default 0.0).
    frequency_penalty : float
        Frequency penalty — proportional to how often token appeared (default 0.0).
    mirostat_mode : int
        Mirostat sampling: 0 = off, 1 = Mirostat, 2 = Mirostat v2 (default 0).
    mirostat_tau : float
        Mirostat target entropy (default 5.0).
    mirostat_eta : float
        Mirostat learning rate (default 0.1).
    seed : int
        RNG seed for reproducible sampling (default: random).

    White-box-only configuration
    ----------------------------
    top_logprobs : int
        Top-K token candidates to return per output position (default 5, max 20).
    logit_bias_tokens : dict[str, float]
        Map of string → logit bias applied to every token the string tokenises
        into.  Use large negative values to suppress tokens (e.g. ``-100`` to
        effectively ban them) or positive values to boost them.
        Example — suppress common refusal openings::

            logit_bias_tokens:
              "I cannot": -100.0
              "I'm sorry": -100.0
              "sorry":     -80.0
    lora_path : str
        Path to a LoRA adapter file (.gguf) to load on top of the base model.
        Loading a custom LoRA is a quintessentially white-box capability.
    lora_scale : float
        LoRA adapter weight scale (default 1.0).
    """

    # Default values kept as class attributes for readability.
    _DEFAULTS: Dict[str, Any] = dict(
        system_prompt="You are a helpful assistant.",
        max_tokens=256,
        temperature=0.7,
        n_ctx=2048,
        n_batch=512,
        n_gpu_layers=0,
        n_threads=None,          # None → auto-detect
        n_threads_batch=None,
        flash_attn=False,
        offload_kqv=True,
        verbose=False,
        # sampling
        top_p=0.95,
        top_k=40,
        min_p=0.05,
        typical_p=1.0,
        tfs_z=1.0,
        repeat_penalty=1.1,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        mirostat_mode=0,
        mirostat_tau=5.0,
        mirostat_eta=0.1,
        seed=None,
        # white-box
        top_logprobs=5,
        lora_path=None,
        lora_scale=1.0,
    )

    def __init__(
        self,
        model_id: str = "llama-cpp-model",
        config: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model_id=model_id, config=config)
        self._llm: Llama | None = None
        self._model_path: Path | None = None
        self._load_error: str | None = None
        self._inference_mode: InferenceMode = InferenceMode.BLACK_BOX
        self._resolved_logit_bias: Dict[int, float] = {}

        # Copy defaults into instance attributes.
        for k, v in self._DEFAULTS.items():
            setattr(self, f"_{k}", v)

        if self.config:
            self._try_configure(self.config)

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def _try_configure(self, params: Dict[str, Any]) -> None:
        model_path_raw = params.get("model_path")
        if not model_path_raw:
            self._load_error = "llama-cpp adapter requires 'model_path' in model.config"
            return

        self._model_path = Path(str(model_path_raw)).expanduser()

        # Inference mode
        raw_mode = str(params.get("inference_mode", "black_box")).lower()
        self._inference_mode = (
            InferenceMode.WHITE_BOX if raw_mode == "white_box" else InferenceMode.BLACK_BOX
        )

        # Apply all scalar config keys using the defaults table as reference.
        for key in self._DEFAULTS:
            raw = params.get(key)
            if raw is None:
                continue
            default = self._DEFAULTS[key]
            if isinstance(default, bool):
                setattr(self, f"_{key}", bool(raw))
            elif isinstance(default, int):
                setattr(self, f"_{key}", int(raw))
            elif isinstance(default, float):
                setattr(self, f"_{key}", float(raw))
            else:
                setattr(self, f"_{key}", raw)

        self._top_logprobs = min(20, max(1, int(self._top_logprobs)))  # type: ignore[attr-defined]

        # LoRA paths
        lora_raw = params.get("lora_path")
        self._lora_path: Optional[Path] = Path(str(lora_raw)).expanduser() if lora_raw else None

        if not self._model_path.exists():
            self._load_error = f"llama-cpp adapter model not found: {self._model_path}"
            self._llm = None
            self._configured = True
            return

        try:
            kwargs: Dict[str, Any] = dict(
                model_path=str(self._model_path),
                n_ctx=self._n_ctx,  # type: ignore[attr-defined]
                n_batch=self._n_batch,  # type: ignore[attr-defined]
                n_gpu_layers=self._n_gpu_layers,  # type: ignore[attr-defined]
                flash_attn=self._flash_attn,  # type: ignore[attr-defined]
                offload_kqv=self._offload_kqv,  # type: ignore[attr-defined]
                verbose=self._verbose,  # type: ignore[attr-defined]
            )
            if self._n_threads is not None:  # type: ignore[attr-defined]
                kwargs["n_threads"] = int(self._n_threads)  # type: ignore[attr-defined]
            if self._n_threads_batch is not None:  # type: ignore[attr-defined]
                kwargs["n_threads_batch"] = int(self._n_threads_batch)  # type: ignore[attr-defined]
            if self._lora_path and self._lora_path.exists():
                kwargs["lora_path"] = str(self._lora_path)
                kwargs["lora_scale"] = float(self._lora_scale)  # type: ignore[attr-defined]

            self._llm = Llama(**kwargs)
            self._load_error = None

            # Resolve logit-bias token strings → token IDs now that the
            # tokenizer is available.
            raw_bias: Dict[str, float] = params.get("logit_bias_tokens", {}) or {}
            self._resolved_logit_bias = self._resolve_logit_bias(raw_bias)

        except Exception as exc:  # pragma: no cover
            self._llm = None
            self._load_error = str(exc)

        self._configured = True

    def _resolve_logit_bias(self, raw_bias: Dict[str, Any]) -> Dict[int, float]:
        """Tokenise string keys into per-token-ID float biases.

        Supports two key formats:
        * Plain integer string (``"42"``) — used as the token ID directly.
        * Multi-character string (``"I cannot"``) — tokenised; the bias is
          applied to every resulting token ID.
        """
        if not raw_bias or self._llm is None:
            return {}
        result: Dict[int, float] = {}
        for key, bias in raw_bias.items():
            try:
                result[int(key)] = float(bias)
            except ValueError:
                token_ids = self._llm.tokenize(key.encode("utf-8"), add_bos=False)
                for tid in token_ids:
                    result[tid] = float(bias)
        return result

    def configure(self, params: Dict[str, Any]) -> None:
        super().configure(params)
        self._try_configure(self.config)

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    async def generate(self, prompt: str, config: Dict[str, Any] | None = None) -> ModelResponse:
        if config:
            self._try_configure({**self.config, **config})
        elif not self._configured:
            self._try_configure(self.config)

        if self._llm is None:
            message = "llama.cpp adapter unavailable"
            if self._load_error:
                message = f"{message}: {self._load_error}"
            return ModelResponse(
                model_id=self.model_id,
                text=f"[Error: {message}]",
                tokens=0,
                latency_ms=0.0,
                metadata={"adapter": "llama-cpp", "error": message},
            )

        white_box = self._inference_mode == InferenceMode.WHITE_BOX

        call_kwargs: Dict[str, Any] = dict(
            messages=[
                {"role": "system", "content": self._system_prompt},  # type: ignore[attr-defined]
                {"role": "user", "content": prompt},
            ],
            # Sampling parameters — passed in both modes.
            temperature=self._temperature,  # type: ignore[attr-defined]
            top_p=self._top_p,  # type: ignore[attr-defined]
            top_k=self._top_k,  # type: ignore[attr-defined]
            min_p=self._min_p,  # type: ignore[attr-defined]
            typical_p=self._typical_p,  # type: ignore[attr-defined]
            tfs_z=self._tfs_z,  # type: ignore[attr-defined]
            repeat_penalty=self._repeat_penalty,  # type: ignore[attr-defined]
            presence_penalty=self._presence_penalty,  # type: ignore[attr-defined]
            frequency_penalty=self._frequency_penalty,  # type: ignore[attr-defined]
            mirostat_mode=self._mirostat_mode,  # type: ignore[attr-defined]
            mirostat_tau=self._mirostat_tau,  # type: ignore[attr-defined]
            mirostat_eta=self._mirostat_eta,  # type: ignore[attr-defined]
            max_tokens=self._max_tokens,  # type: ignore[attr-defined]
        )
        if self._seed is not None:  # type: ignore[attr-defined]
            call_kwargs["seed"] = int(self._seed)  # type: ignore[attr-defined]
        if self._resolved_logit_bias:
            call_kwargs["logit_bias"] = self._resolved_logit_bias
        if white_box:
            call_kwargs["logprobs"] = True
            call_kwargs["top_logprobs"] = self._top_logprobs  # type: ignore[attr-defined]

        started = time.perf_counter()
        chat_result = await asyncio.to_thread(
            self._llm.create_chat_completion, **call_kwargs
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        choice = chat_result.get("choices", [{}])[0]
        message_obj = choice.get("message", {}) if isinstance(choice, dict) else {}
        text = str(message_obj.get("content", "")).strip()

        usage = chat_result.get("usage", {})
        tokens = next(
            (usage.get(k) for k in ("total_tokens", "completion_tokens") if isinstance(usage.get(k), int)),
            len(text.split()),
        )

        # White-box analytics
        logprobs: List[Dict[str, Any]] = []
        perplexity: Optional[float] = None
        token_entropies: List[float] = []
        mean_entropy: Optional[float] = None
        top1_probs: List[float] = []

        if white_box:
            raw_lp = choice.get("logprobs") if isinstance(choice, dict) else None
            logprobs = _extract_logprobs(raw_lp)
            perplexity = _compute_perplexity(logprobs)
            token_entropies = _compute_token_entropies(logprobs)
            mean_entropy = round(sum(token_entropies) / len(token_entropies), 6) if token_entropies else None
            top1_probs = _compute_top1_probs(logprobs)

        return ModelResponse(
            model_id=self.model_id,
            text=text,
            tokens=tokens,
            latency_ms=round(elapsed_ms, 2),
            metadata={
                "adapter": "llama-cpp",
                "inference_mode": self._inference_mode.value,
                "finish_reason": choice.get("finish_reason", "") if isinstance(choice, dict) else "",
                "usage": usage if isinstance(usage, dict) else {},
                "logit_bias_tokens_count": len(self._resolved_logit_bias),
                "lora_loaded": self._lora_path is not None,
            },
            logprobs=logprobs,
            perplexity=perplexity,
            token_entropies=token_entropies,
            mean_entropy=mean_entropy,
            top1_probs=top1_probs,
        )

    # -------------------------------------------------------------------------
    # Health / diagnostics
    # -------------------------------------------------------------------------

    async def health_check(self) -> HealthStatus:
        if self._llm is not None:
            return HealthStatus.OK
        if self._load_error:
            return HealthStatus.DEGRADED
        return HealthStatus.UNAVAILABLE

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "model_path": str(self._model_path) if self._model_path else "",
            "inference_mode": self._inference_mode.value,
            "top_logprobs": self._top_logprobs,  # type: ignore[attr-defined]
            "logit_bias_tokens_count": len(self._resolved_logit_bias),
            "lora_path": str(self._lora_path) if self._lora_path else "",
            "loaded": self._llm is not None,
            "load_error": self._load_error or "",
        }

    async def cost_estimate(self, prompt: str, config: Dict[str, Any] | None = None) -> CostInfo:
        estimated_tokens = max(1, len(prompt.split())) + self._max_tokens  # type: ignore[attr-defined]
        return CostInfo(estimated_tokens=estimated_tokens, estimated_cost_usd=0.0)

    def supported_modes(self) -> Set[InferenceMode]:
        return {InferenceMode.BLACK_BOX, InferenceMode.WHITE_BOX}

    def supports_quantisation(self) -> bool:
        return True


# =============================================================================
# White-box analytics helpers
# =============================================================================

def _extract_logprobs(raw: Any) -> List[Dict[str, Any]]:
    """Normalise the llama-cpp logprobs structure into a flat list.

    Each entry::

        {
          "token":       str,
          "logprob":     float | None,
          "top_logprobs": [{"token": str, "logprob": float}, ...]
        }
    """
    if not isinstance(raw, dict):
        return []
    content = raw.get("content")
    if not isinstance(content, list):
        return []
    result: List[Dict[str, Any]] = []
    for entry in content:
        if not isinstance(entry, dict):
            continue
        top = entry.get("top_logprobs") or []
        result.append({
            "token": entry.get("token", ""),
            "logprob": entry.get("logprob"),
            "top_logprobs": [
                {"token": t.get("token", ""), "logprob": t.get("logprob")}
                for t in top
                if isinstance(t, dict)
            ],
        })
    return result


def _compute_perplexity(logprobs: List[Dict[str, Any]]) -> Optional[float]:
    """Sequence perplexity: exp( −1/N · Σ logprob_i ).

    Returns None when no valid logprobs are present.
    """
    valid = [e["logprob"] for e in logprobs if isinstance(e.get("logprob"), (int, float))]
    if not valid:
        return None
    avg_nll = -sum(valid) / len(valid)
    return round(math.exp(avg_nll), 4)


def _compute_token_entropies(logprobs: List[Dict[str, Any]]) -> List[float]:
    """Per-token Shannon entropy estimated from the top-k candidates.

    The top-k log-probabilities are re-normalised before computing entropy,
    so the result is the conditional entropy over those candidates (an
    approximation — true entropy is higher because probability mass outside
    the top-k is ignored).

    Returns an empty list when no top_logprobs are available.
    """
    entropies: List[float] = []
    for entry in logprobs:
        top = entry.get("top_logprobs") or []
        lps = [t["logprob"] for t in top if isinstance(t.get("logprob"), (int, float))]
        if not lps:
            continue
        probs = [math.exp(lp) for lp in lps]
        total = sum(probs)
        if total == 0:
            entropies.append(0.0)
            continue
        probs = [p / total for p in probs]
        h = -sum(p * math.log(p + 1e-12) for p in probs)
        entropies.append(round(h, 6))
    return entropies


def _compute_top1_probs(logprobs: List[Dict[str, Any]]) -> List[float]:
    """Probability of the sampled token at each output position.

    Computed as exp(logprob) of the chosen token — a direct measure of how
    'confidently' the model generated each token.
    """
    result: List[float] = []
    for entry in logprobs:
        lp = entry.get("logprob")
        result.append(round(math.exp(lp), 6) if isinstance(lp, (int, float)) else 0.0)
    return result
