from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class LanguageVocabulary:
    """A language-specific view of the shared multilingual vocabulary."""

    tokens: tuple[str, ...]
    allowed_ids: frozenset[int]
    blank_id: int
    unknown_id: int | None = None

    def token(self, token_id: int) -> str:
        if token_id < 0 or token_id >= len(self.tokens):
            raise ValueError(f"token id {token_id} is outside vocabulary size {len(self.tokens)}")
        return self.tokens[token_id]


def load_language_vocabularies(
    path: Path, languages: Sequence[str]
) -> dict[str, LanguageVocabulary]:
    """Load an explicit multilingual token table and per-language masks."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse language vocabulary asset {path}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise ValueError(f"{path} must contain a JSON object")

    global_tokens = _find_tokens(document)
    language_nodes = _language_nodes(document)
    result: dict[str, LanguageVocabulary] = {}
    for language in languages:
        node = language_nodes.get(language)
        if node is None:
            raise ValueError(f"{path} has no vocabulary/mask for language {language!r}")

        local_tokens = _find_tokens(node) if isinstance(node, Mapping) else None
        tokens = local_tokens or global_tokens
        direct_mapping = _token_id_mapping(node)
        if tokens is None and direct_mapping is not None:
            tokens = _tokens_from_mapping(direct_mapping)
        if tokens is None:
            raise ValueError(f"{path} language {language!r} has a mask but no token vocabulary")

        allowed = _find_allowed_ids(node)
        if allowed is None and direct_mapping is not None:
            allowed = frozenset(direct_mapping.values())
        if allowed is None:
            if local_tokens is not None:
                allowed = frozenset(range(len(tokens)))
            else:
                raise ValueError(f"{path} language {language!r} does not declare allowed token ids")
        if not allowed:
            raise ValueError(f"{path} language {language!r} has an empty token mask")
        invalid = [token_id for token_id in allowed if token_id < 0 or token_id >= len(tokens)]
        if invalid:
            raise ValueError(
                f"{path} language {language!r} mask contains out-of-range id {invalid[0]} "
                f"for vocabulary size {len(tokens)}"
            )

        blank_id = _special_id(node, document, ("blank_id", "blank", "ctc_blank_id"))
        if blank_id is None:
            blank_id = _token_named(tokens, ("<blank>", "<blk>", "[BLANK]"))
        if blank_id is None:
            blank_id = len(tokens) - 1
        if blank_id < 0 or blank_id >= len(tokens):
            raise ValueError(f"{path} language {language!r} blank id {blank_id} is out of range")
        unknown_id = _special_id(node, document, ("unknown_id", "unk_id", "unk"))
        if unknown_id is None:
            unknown_id = _token_named(tokens, ("<unk>", "[UNK]"))

        result[language] = LanguageVocabulary(
            tokens=tokens,
            allowed_ids=allowed | {blank_id},
            blank_id=blank_id,
            unknown_id=unknown_id,
        )
    return result


class CTCGreedyDecoder:
    """Deterministic masked greedy CTC decoding for terminal ONNX output."""

    def __init__(self, vocabularies: Mapping[str, LanguageVocabulary]) -> None:
        self._vocabularies = dict(vocabularies)

    def decode_batch(
        self,
        scores_or_ids: npt.NDArray[np.generic],
        languages: Sequence[str],
        lengths: npt.NDArray[np.integer[Any]] | None = None,
    ) -> list[str]:
        values = np.asarray(scores_or_ids)
        if values.ndim not in (2, 3):
            raise ValueError(
                f"CTC output must have shape [batch,time] or [batch,time,vocab], got {values.shape}"
            )
        if values.shape[0] != len(languages):
            raise ValueError(
                f"CTC batch size {values.shape[0]} does not match {len(languages)} languages"
            )
        flat_lengths = None if lengths is None else np.asarray(lengths).reshape(-1)
        if flat_lengths is not None and flat_lengths.size != len(languages):
            raise ValueError("CTC output-length count does not match batch size")

        decoded: list[str] = []
        for index, language in enumerate(languages):
            vocabulary = self._vocabularies.get(language)
            if vocabulary is None:
                raise ValueError(f"unsupported CTC language {language!r}")
            time_steps = values.shape[1]
            if flat_lengths is not None:
                time_steps = min(time_steps, max(0, int(flat_lengths[index])))
            item = values[index, :time_steps]
            if values.ndim == 3:
                item = self._masked_argmax(item, vocabulary)
            decoded.append(self._decode_ids(np.asarray(item).reshape(-1), vocabulary))
        return decoded

    @staticmethod
    def _masked_argmax(
        scores: npt.NDArray[np.generic], vocabulary: LanguageVocabulary
    ) -> npt.NDArray[np.int64]:
        if scores.shape[-1] != len(vocabulary.tokens):
            raise ValueError(
                f"CTC graph emitted {scores.shape[-1]} classes but language vocabulary has "
                f"{len(vocabulary.tokens)} tokens"
            )
        allowed = np.fromiter(vocabulary.allowed_ids, dtype=np.int64)
        local = np.argmax(scores[:, allowed], axis=-1)
        return np.asarray(allowed[local], dtype=np.int64)

    @staticmethod
    def _decode_ids(token_ids: npt.NDArray[np.generic], vocabulary: LanguageVocabulary) -> str:
        pieces: list[str] = []
        previous: int | None = None
        for raw_id in token_ids:
            token_id = int(raw_id)
            if token_id == previous:
                continue
            previous = token_id
            if token_id == vocabulary.blank_id:
                continue
            if token_id not in vocabulary.allowed_ids:
                raise ValueError(f"CTC graph emitted token {token_id} outside the language mask")
            if vocabulary.unknown_id is not None and token_id == vocabulary.unknown_id:
                continue
            token = vocabulary.token(token_id)
            if token.startswith("<") and token.endswith(">"):
                continue
            pieces.append(token)
        return _join_pieces(pieces)


def tokens_to_text(token_ids: Sequence[int], vocabulary: LanguageVocabulary) -> str:
    pieces: list[str] = []
    for token_id in token_ids:
        if token_id == vocabulary.blank_id:
            continue
        if token_id not in vocabulary.allowed_ids:
            raise ValueError(f"RNNT graph emitted token {token_id} outside the language mask")
        if vocabulary.unknown_id is not None and token_id == vocabulary.unknown_id:
            continue
        token = vocabulary.token(token_id)
        if token.startswith("<") and token.endswith(">"):
            continue
        pieces.append(token)
    return _join_pieces(pieces)


def _join_pieces(pieces: Sequence[str]) -> str:
    text = "".join(pieces).replace("▁", " ")
    text = text.replace("@@ ", "").replace("@@", "")
    return " ".join(text.split())


def _language_nodes(document: Mapping[str, object]) -> Mapping[str, object]:
    for key in ("languages", "masks", "language_masks", "vocabularies"):
        value = document.get(key)
        if isinstance(value, Mapping):
            return value
    return document


def _find_tokens(node: Mapping[str, object]) -> tuple[str, ...] | None:
    for key in ("tokens", "vocab", "vocabulary", "labels"):
        value = node.get(key)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return tuple(value)
        mapping = _token_id_mapping(value)
        if mapping is not None:
            return _tokens_from_mapping(mapping)
    direct = _token_id_mapping(node)
    if direct is not None:
        return _tokens_from_mapping(direct)
    return None


def _token_id_mapping(node: object) -> dict[str, int] | None:
    if not isinstance(node, Mapping) or not node:
        return None
    if all(isinstance(key, str) and isinstance(value, int) for key, value in node.items()):
        return {key: value for key, value in node.items()}
    if all(
        isinstance(key, str) and key.isdecimal() and isinstance(value, str)
        for key, value in node.items()
    ):
        return {value: int(key) for key, value in node.items()}
    return None


def _tokens_from_mapping(mapping: Mapping[str, int]) -> tuple[str, ...]:
    if not mapping:
        raise ValueError("token vocabulary is empty")
    largest = max(mapping.values())
    if min(mapping.values()) < 0 or set(mapping.values()) != set(range(largest + 1)):
        raise ValueError("token vocabulary ids must be contiguous and start at zero")
    result = [""] * (largest + 1)
    for token, token_id in mapping.items():
        result[token_id] = token
    return tuple(result)


def _find_allowed_ids(node: object) -> frozenset[int] | None:
    if isinstance(node, list) and all(isinstance(value, int) for value in node):
        return frozenset(node)
    if not isinstance(node, Mapping):
        return None
    for key in ("allowed_ids", "token_ids", "mask", "ids"):
        value = node.get(key)
        if isinstance(value, list) and all(isinstance(item, int) for item in value):
            if value and set(value) <= {0, 1} and len(value) > 2:
                return frozenset(index for index, enabled in enumerate(value) if enabled)
            return frozenset(value)
        mapping = _token_id_mapping(value)
        if mapping is not None:
            return frozenset(mapping.values())
    return None


def _special_id(local: object, root: Mapping[str, object], names: Sequence[str]) -> int | None:
    for node in (local, root):
        if isinstance(node, Mapping):
            for name in names:
                value = node.get(name)
                if isinstance(value, int):
                    return value
    return None


def _token_named(tokens: Sequence[str], names: Sequence[str]) -> int | None:
    for name in names:
        try:
            return tokens.index(name)
        except ValueError:
            pass
    return None
