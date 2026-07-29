from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from app.engine.ctc_decoder import LanguageVocabulary, tokens_to_text
from app.engine.errors import DecodeLimitError

ScoreFunction = Callable[[int, tuple[int, ...]], npt.NDArray[np.floating]]


@dataclass(frozen=True, slots=True)
class RNNTDecodeLimits:
    """Hard work bounds for the intentionally eager greedy decoder."""

    max_symbols_per_frame: int = 5
    max_total_symbols: int = 4096

    def __post_init__(self) -> None:
        if self.max_symbols_per_frame <= 0:
            raise ValueError("max_symbols_per_frame must be positive")
        if self.max_total_symbols <= 0:
            raise ValueError("max_total_symbols must be positive")


class BoundedRNNTGreedyDecoder:
    """Bounded eager RNNT greedy search.

    ``score`` is invoked synchronously for every prediction. This is deliberately
    not described as compiled, streaming, or cache-aware: exported graph metadata
    must make such capabilities explicit before they can be implemented safely.
    """

    def __init__(
        self,
        vocabularies: Mapping[str, LanguageVocabulary],
        limits: RNNTDecodeLimits | None = None,
    ) -> None:
        self._vocabularies = dict(vocabularies)
        self._limits = limits or RNNTDecodeLimits()

    def decode(
        self,
        *,
        language: str,
        frame_count: int,
        score: ScoreFunction,
    ) -> str:
        vocabulary = self._vocabularies.get(language)
        if vocabulary is None:
            raise ValueError(f"unsupported RNNT language {language!r}")
        if frame_count < 0:
            raise ValueError("frame_count cannot be negative")

        emitted: list[int] = []
        for frame_index in range(frame_count):
            for _ in range(self._limits.max_symbols_per_frame):
                logits = np.asarray(score(frame_index, tuple(emitted)))
                if logits.ndim == 0:
                    raise ValueError("RNNT joint graph returned a scalar instead of token scores")
                flattened = logits.reshape(-1, logits.shape[-1])
                if flattened.shape[0] != 1:
                    raise ValueError(
                        "RNNT greedy score must contain exactly one prediction position; "
                        f"got shape {logits.shape}"
                    )
                if flattened.shape[-1] != len(vocabulary.tokens):
                    raise ValueError(
                        f"RNNT graph emitted {flattened.shape[-1]} classes but language "
                        f"vocabulary has {len(vocabulary.tokens)} tokens"
                    )
                allowed = np.fromiter(vocabulary.allowed_ids, dtype=np.int64)
                token_id = int(allowed[int(np.argmax(flattened[0, allowed]))])
                if token_id == vocabulary.blank_id:
                    break
                emitted.append(token_id)
                if len(emitted) >= self._limits.max_total_symbols:
                    raise DecodeLimitError(
                        f"RNNT emitted {len(emitted)} symbols without completing; "
                        f"hard limit is {self._limits.max_total_symbols}"
                    )
        return tokens_to_text(emitted, vocabulary)

    def decode_many(
        self,
        *,
        language: str,
        frame_counts: Sequence[int],
        scores: Sequence[ScoreFunction],
    ) -> list[str]:
        if len(frame_counts) != len(scores):
            raise ValueError("RNNT frame-count and score callback counts differ")
        return [
            self.decode(language=language, frame_count=frame_count, score=score)
            for frame_count, score in zip(frame_counts, scores, strict=True)
        ]
