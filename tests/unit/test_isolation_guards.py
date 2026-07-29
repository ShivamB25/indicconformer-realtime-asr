"""The suite's own isolation guarantees, asserted rather than assumed.

If the autouse fixtures in ``tests/conftest.py`` are ever weakened, these tests
fail instead of the suite quietly gaining the ability to reach the network or to
read a developer's ``ASR_*`` environment.
"""

from __future__ import annotations

import os
import socket

import pytest

from app.core.config import get_settings
from app.core.types import EngineKind


class TestNetworkIsGuarded:
    def test_connecting_a_socket_is_refused(self) -> None:
        with pytest.raises(AssertionError, match="must not use the network"):
            socket.socket().connect(("127.0.0.1", 9))

    def test_the_nonblocking_connect_variant_is_refused(self) -> None:
        with pytest.raises(AssertionError, match="must not use the network"):
            socket.socket().connect_ex(("127.0.0.1", 9))

    def test_the_convenience_constructor_is_refused(self) -> None:
        with pytest.raises(AssertionError, match="must not use the network"):
            socket.create_connection(("127.0.0.1", 9))

    def test_name_resolution_is_refused(self) -> None:
        with pytest.raises(AssertionError, match="must not use the network"):
            socket.getaddrinfo("huggingface.co", 443)


class TestEnvironmentIsGuarded:
    def test_no_ambient_asr_configuration_leaks_into_tests(self) -> None:
        assert {name for name in os.environ if name.startswith("ASR_")} == {
            "ASR_ENGINE",
            "ASR_ENVIRONMENT",
        }

    def test_the_offline_flags_are_set(self) -> None:
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"

    def test_no_cuda_device_is_visible(self) -> None:
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""

    def test_process_settings_select_the_mock_engine(self) -> None:
        settings = get_settings()

        assert settings.engine is EngineKind.MOCK
        assert settings.environment == "test"
        assert settings.offline is True

    def test_settings_are_not_cached_across_tests(self) -> None:
        """Each test re-reads configuration, so ordering cannot change results."""

        assert get_settings.cache_info().currsize <= 1
