"""Contract tests for the ingestion-boundary resampler.

Linear interpolation is the documented, dependency-free choice, so its exact
output is pinned here: a change of algorithm must be a deliberate decision, not
a silent one.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.audio.pcm import SAMPLE_RATE
from app.audio.resample import ResampleError, resample_audio


class TestPassThrough:
    def test_matching_rates_only_convert_the_dtype(self) -> None:
        audio = np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32)
        result = resample_audio(audio, SAMPLE_RATE, SAMPLE_RATE)
        assert result.dtype == np.float32
        assert np.array_equal(result, audio)

    def test_the_input_array_is_never_aliased(self) -> None:
        audio = np.array([0.25, 0.5], dtype=np.float32)
        result = resample_audio(audio, SAMPLE_RATE, SAMPLE_RATE)
        result[0] = -1.0
        assert audio[0] == pytest.approx(0.25)

    def test_the_default_target_rate_is_the_service_rate(self) -> None:
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        assert resample_audio(audio, SAMPLE_RATE).size == SAMPLE_RATE

    def test_empty_audio_stays_empty(self) -> None:
        result = resample_audio(np.array([], dtype=np.float32), 8_000, SAMPLE_RATE)
        assert result.size == 0
        assert result.dtype == np.float32


class TestRateConversion:
    @pytest.mark.parametrize(
        ("source_rate", "input_size", "expected_size"),
        [
            (8_000, 160, 320),
            (32_000, 320, 160),
            (48_000, 480, 160),
            (44_100, 441, 160),
            (22_050, 441, 320),
        ],
    )
    def test_output_length_follows_the_rate_ratio(
        self, source_rate: int, input_size: int, expected_size: int
    ) -> None:
        audio = np.zeros(input_size, dtype=np.float32)
        assert resample_audio(audio, source_rate).size == expected_size

    def test_a_single_sample_is_never_dropped(self) -> None:
        result = resample_audio(np.array([0.5], dtype=np.float32), 48_000, SAMPLE_RATE)
        assert result.size == 1
        assert result[0] == pytest.approx(0.5)

    def test_linear_interpolation_is_exact(self) -> None:
        audio = np.array([0.0, 1.0], dtype=np.float32)
        result = resample_audio(audio, 1, 2)
        assert result.size == 4
        assert result.tolist() == pytest.approx([0.0, 0.5, 1.0, 1.0])

    def test_downsampling_picks_the_interpolated_positions(self) -> None:
        audio = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        result = resample_audio(audio, 2, 1)
        assert result.tolist() == pytest.approx([0.0, 2.0])

    def test_a_constant_signal_survives_resampling(self) -> None:
        audio = np.full(1_000, 0.3, dtype=np.float32)
        result = resample_audio(audio, 44_100, SAMPLE_RATE)
        assert np.allclose(result, 0.3)

    def test_output_is_contiguous_float32(self) -> None:
        result = resample_audio(np.zeros(441, dtype=np.float32), 44_100)
        assert result.dtype == np.float32
        assert result.flags["C_CONTIGUOUS"]

    def test_resampling_is_deterministic(self) -> None:
        audio = np.linspace(-1.0, 1.0, 999, dtype=np.float32)
        first = resample_audio(audio, 44_100)
        second = resample_audio(audio, 44_100)
        assert np.array_equal(first, second)


class TestIntegerInput:
    def test_int16_is_normalized_by_full_scale(self) -> None:
        audio = np.array([-32_768, 0, 32_767], dtype=np.int16)
        result = resample_audio(audio, SAMPLE_RATE)
        assert result.dtype == np.float32
        assert result[0] == pytest.approx(-1.0)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(32_767 / 32_768)

    @pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32])
    def test_every_integer_width_lands_inside_the_unit_range(
        self, dtype: type[np.signedinteger[np.typing.NBitBase]]
    ) -> None:
        info = np.iinfo(dtype)
        audio = np.array([info.min, info.max], dtype=dtype)
        result = resample_audio(audio, SAMPLE_RATE)
        assert float(np.abs(result).max()) <= 1.0

    def test_integer_input_is_not_modified(self) -> None:
        audio = np.array([1_000, 2_000], dtype=np.int16)
        original = audio.copy()
        resample_audio(audio, 8_000)
        assert np.array_equal(audio, original)


class TestRejectedInput:
    @pytest.mark.parametrize(
        ("source_rate", "target_rate"),
        [(0, SAMPLE_RATE), (-8_000, SAMPLE_RATE), (SAMPLE_RATE, 0), (SAMPLE_RATE, -1)],
    )
    def test_non_positive_rates_are_refused(self, source_rate: int, target_rate: int) -> None:
        audio = np.zeros(10, dtype=np.float32)
        with pytest.raises(ResampleError, match="sample rates must be positive"):
            resample_audio(audio, source_rate, target_rate)

    def test_multichannel_audio_is_refused(self) -> None:
        with pytest.raises(ResampleError, match="one-dimensional mono"):
            resample_audio(np.zeros((2, 100), dtype=np.float32), SAMPLE_RATE)

    def test_non_numeric_audio_is_refused(self) -> None:
        with pytest.raises(ResampleError, match="numeric dtype"):
            resample_audio(np.array(["a", "b"]), SAMPLE_RATE)

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
    def test_non_finite_audio_is_refused(self, bad_value: float) -> None:
        audio = np.array([0.0, bad_value, 0.2], dtype=np.float32)
        with pytest.raises(ResampleError, match="non-finite"):
            resample_audio(audio, SAMPLE_RATE)

    def test_resample_errors_are_value_errors(self) -> None:
        assert issubclass(ResampleError, ValueError)
