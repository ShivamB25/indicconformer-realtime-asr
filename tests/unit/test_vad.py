"""Contract tests for the deterministic energy VAD.

Frames here are built so their RMS is exact: an alternating +/-A int16 frame has
RMS A/32768, and a constant float frame has RMS equal to that constant. That
makes the decision boundary testable without tolerances.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.audio.pcm import SAMPLES_PER_FRAME
from app.audio.vad import EnergyVAD, EnergyVADConfig, VADDecision
from tests.support.audio import float_frame, int16_frame

DEFAULT_THRESHOLD = 0.015


class TestConfiguration:
    def test_default_threshold_is_the_documented_one(self) -> None:
        assert EnergyVADConfig().speech_threshold == pytest.approx(DEFAULT_THRESHOLD)

    @pytest.mark.parametrize("threshold", [0.0, -0.1, 1.01, 2.0])
    def test_threshold_must_be_a_normalized_fraction(self, threshold: float) -> None:
        with pytest.raises(ValueError, match=r"speech_threshold must be in \(0, 1\]"):
            EnergyVADConfig(speech_threshold=threshold)

    def test_full_scale_threshold_is_allowed(self) -> None:
        assert EnergyVADConfig(speech_threshold=1.0).speech_threshold == 1.0

    def test_the_default_config_is_used_when_none_is_supplied(self) -> None:
        assert EnergyVAD().config == EnergyVADConfig()


class TestDecisionBoundary:
    def test_energy_exactly_at_the_threshold_counts_as_speech(self) -> None:
        """The comparison is inclusive, so the threshold itself is speech."""

        exactly_at_threshold = np.full(SAMPLES_PER_FRAME, DEFAULT_THRESHOLD, dtype=np.float64)
        decision = EnergyVAD().classify(exactly_at_threshold)
        assert decision.rms == DEFAULT_THRESHOLD
        assert decision.is_speech is True

    def test_energy_just_below_the_threshold_is_not_speech(self) -> None:
        vad = EnergyVAD()
        assert vad.classify(float_frame(DEFAULT_THRESHOLD - 1e-4)).is_speech is False

    def test_digital_silence_is_not_speech(self) -> None:
        decision = EnergyVAD().classify(np.zeros(SAMPLES_PER_FRAME, dtype=np.int16))
        assert decision.is_speech is False
        assert decision.rms == 0.0

    def test_a_custom_threshold_moves_the_boundary(self) -> None:
        frame = float_frame(0.05)
        assert EnergyVAD(EnergyVADConfig(speech_threshold=0.04)).is_speech(frame) is True
        assert EnergyVAD(EnergyVADConfig(speech_threshold=0.06)).is_speech(frame) is False

    def test_is_speech_agrees_with_classify(self) -> None:
        vad = EnergyVAD()
        for amplitude in (0, 100, 500, 6_000, 32_767):
            frame = int16_frame(amplitude)
            assert vad.is_speech(frame) is vad.classify(frame).is_speech


class TestEnergyMeasurement:
    @pytest.mark.parametrize("amplitude", [0, 1, 491, 492, 6_000, 32_767])
    def test_integer_frames_are_normalized_by_the_dtype_peak(self, amplitude: int) -> None:
        decision = EnergyVAD().classify(int16_frame(amplitude))
        assert decision.rms == pytest.approx(amplitude / 32_768.0, abs=1e-12)

    def test_integer_and_float_frames_measure_the_same_energy(self) -> None:
        vad = EnergyVAD()
        integer = vad.classify(int16_frame(6_000))
        floating = vad.classify(float_frame(6_000 / 32_768.0))
        assert integer.rms == pytest.approx(floating.rms)
        assert integer.is_speech == floating.is_speech

    @pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32])
    def test_every_integer_width_is_normalized_to_full_scale(
        self,
        dtype: np.dtype[np.signedinteger[np.typing.NBitBase]]
        | type[np.signedinteger[np.typing.NBitBase]],
    ) -> None:
        info = np.iinfo(dtype)
        frame = np.full(SAMPLES_PER_FRAME, info.max, dtype=dtype)
        decision = EnergyVAD().classify(frame)
        assert decision.rms == pytest.approx(info.max / abs(info.min), abs=1e-9)
        assert decision.is_speech is True

    def test_dbfs_matches_the_measured_rms(self) -> None:
        decision = EnergyVAD().classify(float_frame(0.1))
        assert decision.dbfs == pytest.approx(20.0 * math.log10(0.1))

    def test_silence_reports_a_clamped_floor_instead_of_negative_infinity(self) -> None:
        decision = EnergyVAD().classify(float_frame(0.0))
        assert decision.dbfs == pytest.approx(-240.0)
        assert math.isfinite(decision.dbfs)

    def test_negative_and_positive_energy_are_symmetric(self) -> None:
        vad = EnergyVAD()
        assert vad.classify(float_frame(0.2)).rms == pytest.approx(
            vad.classify(float_frame(-0.2)).rms
        )

    def test_classification_is_repeatable(self) -> None:
        vad = EnergyVAD()
        frame = int16_frame(1_000)
        assert vad.classify(frame) == vad.classify(frame)

    def test_the_decision_is_an_immutable_value(self) -> None:
        decision = EnergyVAD().classify(float_frame(0.2))
        assert isinstance(decision, VADDecision)
        with pytest.raises(AttributeError):
            decision.is_speech = False  # type: ignore[misc]

    def test_classification_does_not_modify_the_frame(self) -> None:
        frame = int16_frame(2_000)
        original = frame.copy()
        EnergyVAD().classify(frame)
        assert np.array_equal(frame, original)


class TestFrameValidation:
    @pytest.mark.parametrize("size", [0, 1, 319, 321, 640])
    def test_only_complete_frames_are_classified(self, size: int) -> None:
        with pytest.raises(ValueError, match="exactly 320 mono samples"):
            EnergyVAD().classify(np.zeros(size, dtype=np.int16))

    def test_multichannel_frames_are_refused(self) -> None:
        stereo = np.zeros((2, SAMPLES_PER_FRAME), dtype=np.int16)
        with pytest.raises(ValueError, match="exactly 320 mono samples"):
            EnergyVAD().classify(stereo)

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
    def test_non_finite_samples_are_refused(self, bad_value: float) -> None:
        frame = float_frame(0.1)
        frame[7] = bad_value
        with pytest.raises(ValueError, match="non-finite"):
            EnergyVAD().classify(frame)

    @pytest.mark.parametrize("dtype", [np.bool_, np.str_])
    def test_non_numeric_dtypes_are_refused(self, dtype: type[np.generic]) -> None:
        frame = np.zeros(SAMPLES_PER_FRAME, dtype=dtype)
        with pytest.raises(ValueError, match="numeric dtype"):
            EnergyVAD().classify(frame)
