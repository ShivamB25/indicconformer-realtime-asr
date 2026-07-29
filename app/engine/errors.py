from __future__ import annotations


class OrtEngineError(RuntimeError):
    """Base class for actionable local-model runtime failures."""


class ManifestVerificationError(OrtEngineError):
    """The pinned local model manifest or one of its files is invalid."""


class AssetDiscoveryError(OrtEngineError):
    """A required model asset was not present in the verified manifest."""


class ProviderUnavailableError(OrtEngineError):
    """The explicitly configured execution provider cannot be used safely."""


class ModelContractError(OrtEngineError):
    """An ONNX graph does not expose the contract required by the engine."""


class EngineNotReadyError(OrtEngineError):
    """Inference was requested before successful engine startup."""


class DecodeLimitError(OrtEngineError):
    """RNNT decoding exceeded its configured hard bound."""
