"""Shared, closed sets used by configuration and API validation."""

from enum import StrEnum


class LanguageCode(StrEnum):
    AS = "as"
    BN = "bn"
    BRX = "brx"
    DOI = "doi"
    GU = "gu"
    HI = "hi"
    KN = "kn"
    KOK = "kok"
    KS = "ks"
    MAI = "mai"
    ML = "ml"
    MNI = "mni"
    MR = "mr"
    NE = "ne"
    OR = "or"
    PA = "pa"
    SA = "sa"
    SAT = "sat"
    SD = "sd"
    TA = "ta"
    TE = "te"
    UR = "ur"


SUPPORTED_LANGUAGES: tuple[str, ...] = tuple(language.value for language in LanguageCode)
SUPPORTED_LANGUAGE_CODES: frozenset[str] = frozenset(SUPPORTED_LANGUAGES)


class ProcessingMode(StrEnum):
    LATENCY = "latency"
    HYBRID = "hybrid"
    ACCURACY = "accuracy"


class Decoder(StrEnum):
    CTC = "ctc"
    RNNT = "rnnt"


class EngineKind(StrEnum):
    MOCK = "mock"
    OFFICIAL = "official"


class VADKind(StrEnum):
    DISABLED = "disabled"
    ENERGY = "energy"
    SILERO = "silero"
    WEBRTC = "webrtc"
