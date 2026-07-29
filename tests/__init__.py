"""CPU-only, deterministic test suite.

Every test in this package must run without a network, a GPU, or model
weights. The only inference backend used anywhere is ``MockEngine`` (or a
test-owned double built on the same public engine contract).
"""
