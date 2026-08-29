"""Contract tests for PCM16 framing and the bounded session buffer.

The realtime protocol accepts exactly one 20 ms mono frame per binary message.
At 16 kHz that is 320 samples and 640 bytes; conflating those two numbers is the
most likely framing bug, so both are asserted explicitly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from app.audio.pcm import (
    BYTES_PER_SAMPLE,
    FRAME_DURATION_MS,
    PCM16_FRAME_BYTES,
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
    PCM16Buffer,
    PCMBufferOverflow,
    PCMError,
    PCMFrameSizeError,
    decode_pcm16_frame,
    pcm16_to_float32,
)
from tests.support.audio import frame_bytes, int16_frame, silence_frame, speech_frame


class TestFrameGeometry:
    def test_frame_constants_describe_one_twenty_millisecond_mono_frame(self) -> None:
        assert SAMPLE_RATE == 16_000
        assert FRAME_DURATION_MS == 20
        assert SAMPLES_PER_FRAME == 320
        assert BYTES_PER_SAMPLE == 2
        assert PCM16_FRAME_BYTES == 640
        assert PCM16_FRAME_BYTES == SAMPLES_PER_FRAME * BYTES_PER_SAMPLE


class TestDecodeFrame:
    def test_exactly_one_frame_is_accepted(self) -> None:
        decoded = decode_pcm16_frame(speech_frame())
        assert decoded.shape == (SAMPLES_PER_FRAME,)
        assert decoded.dtype == np.int16

    @pytest.mark.parametrize(
        "size",
        [0, 1, 2, 319, 320, 321, 639, 641, 642, 1_280],
        ids=[
            "empty",
            "one_byte",
            "one_sample",
            "319_bytes",
            "sample_count_as_byte_count",
            "321_bytes",
            "one_byte_short",
            "one_byte_long",
            "one_sample_long",
            "two_frames",
        ],
    )
    def test_any_other_length_is_rejected(self, size: int) -> None:
        with pytest.raises(PCMFrameSizeError) as caught:
            decode_pcm16_frame(b"\x00" * size)
        assert "expected 640 bytes" in str(caught.value)
        assert str(size) in str(caught.value)

    def test_samples_are_decoded_as_little_endian(self) -> None:
        payload = b"\x01\x02" + b"\x00" * (PCM16_FRAME_BYTES - 2)
        decoded = decode_pcm16_frame(payload)
        assert int(decoded[0]) == 0x0201

    def test_full_scale_values_survive_decoding(self) -> None:
        payload = b"\x00\x80" + b"\xff\x7f" + b"\x00" * (PCM16_FRAME_BYTES - 4)
        decoded = decode_pcm16_frame(payload)
        assert int(decoded[0]) == -32_768
        assert int(decoded[1]) == 32_767

    @pytest.mark.parametrize(
        "wrap", [bytes, bytearray, memoryview], ids=["bytes", "bytearray", "memoryview"]
    )
    def test_any_contiguous_buffer_type_is_accepted(self, wrap: type[Any]) -> None:
        payload = wrap(speech_frame())
        assert decode_pcm16_frame(payload).size == SAMPLES_PER_FRAME

    def test_the_transport_buffer_is_never_retained(self) -> None:
        source = bytearray(speech_frame())
        decoded = decode_pcm16_frame(source)
        original = int(decoded[0])
        source[0:2] = b"\x00\x00"
        assert int(decoded[0]) == original

    def test_non_contiguous_input_is_refused(self) -> None:
        strided = memoryview(bytearray(PCM16_FRAME_BYTES * 2))[::2]
        assert strided.nbytes == PCM16_FRAME_BYTES
        with pytest.raises(PCMError, match="contiguous"):
            decode_pcm16_frame(strided)

    def test_silence_decodes_to_zeros(self) -> None:
        assert not decode_pcm16_frame(silence_frame()).any()


class TestFloatConversion:
    def test_scaling_uses_the_full_negative_scale(self) -> None:
        samples = np.array([-32_768, 0, 32_767], dtype=np.int16)
        converted = pcm16_to_float32(samples)
        assert converted.dtype == np.float32
        assert converted[0] == pytest.approx(-1.0)
        assert converted[1] == pytest.approx(0.0)
        assert converted[2] == pytest.approx(32_767 / 32_768)

    def test_conversion_stays_inside_the_engine_input_range(self) -> None:
        converted = pcm16_to_float32(int16_frame(32_767))
        assert float(np.abs(converted).max()) <= 1.0

    def test_multichannel_audio_is_refused(self) -> None:
        stereo = np.zeros((2, SAMPLES_PER_FRAME), dtype=np.int16)
        with pytest.raises(PCMError, match="mono"):
            pcm16_to_float32(stereo)

    def test_conversion_is_deterministic(self) -> None:
        first = pcm16_to_float32(int16_frame(1_234))
        second = pcm16_to_float32(int16_frame(1_234))
        assert np.array_equal(first, second)


class TestBufferConstruction:
    @pytest.mark.parametrize("duration_ms", [0, 19, -20])
    def test_a_buffer_must_hold_at_least_one_frame(self, duration_ms: int) -> None:
        with pytest.raises(ValueError, match="at least one frame"):
            PCM16Buffer(duration_ms)

    @pytest.mark.parametrize("duration_ms", [21, 30, 199, 1_001])
    def test_capacity_must_be_a_whole_number_of_frames(self, duration_ms: int) -> None:
        with pytest.raises(ValueError, match="multiple of 20 ms"):
            PCM16Buffer(duration_ms)

    def test_a_new_buffer_is_empty(self) -> None:
        buffer = PCM16Buffer(1_000)
        assert buffer.empty is True
        assert buffer.frame_count == 0
        assert buffer.duration_ms == 0
        assert buffer.to_int16().size == 0


class TestBufferAppend:
    def test_duration_tracks_appended_frames(self) -> None:
        buffer = PCM16Buffer(1_000)
        for index in range(1, 6):
            buffer.append(speech_frame())
            assert buffer.frame_count == index
            assert buffer.duration_ms == index * FRAME_DURATION_MS
        assert buffer.empty is False

    @pytest.mark.parametrize("size", [0, 320, 639, 641, 1_280])
    def test_partial_frames_are_never_buffered(self, size: int) -> None:
        buffer = PCM16Buffer(1_000)
        with pytest.raises(PCMFrameSizeError):
            buffer.append(b"\x00" * size)
        assert buffer.frame_count == 0

    def test_capacity_is_enforced_at_the_exact_limit(self) -> None:
        buffer = PCM16Buffer(100)
        for _ in range(5):
            buffer.append(speech_frame())
        assert buffer.duration_ms == 100

        with pytest.raises(PCMBufferOverflow, match="full"):
            buffer.append(speech_frame())

    def test_a_rejected_frame_leaves_the_buffer_unchanged(self) -> None:
        buffer = PCM16Buffer(40)
        buffer.append(frame_bytes(int16_frame(1_000)))
        buffer.append(frame_bytes(int16_frame(2_000)))
        before = buffer.to_int16()

        with pytest.raises(PCMBufferOverflow):
            buffer.append(frame_bytes(int16_frame(3_000)))

        assert buffer.frame_count == 2
        assert np.array_equal(buffer.to_int16(), before)

    def test_clear_frees_the_whole_buffer_for_reuse(self) -> None:
        buffer = PCM16Buffer(40)
        buffer.append(speech_frame())
        buffer.append(speech_frame())
        buffer.clear()
        assert buffer.empty is True
        assert buffer.duration_ms == 0
        buffer.append(speech_frame())
        assert buffer.frame_count == 1


class TestBufferConversion:
    def test_frames_are_concatenated_in_arrival_order(self) -> None:
        buffer = PCM16Buffer(60)
        buffer.append(frame_bytes(int16_frame(100)))
        buffer.append(frame_bytes(int16_frame(200)))
        samples = buffer.to_int16()
        assert samples.size == SAMPLES_PER_FRAME * 2
        assert int(samples[0]) == 100
        assert int(samples[SAMPLES_PER_FRAME]) == 200

    def test_float_conversion_matches_the_standalone_helper(self) -> None:
        buffer = PCM16Buffer(40)
        buffer.append(frame_bytes(int16_frame(4_096)))
        assert np.array_equal(buffer.to_float32(), pcm16_to_float32(buffer.to_int16()))
        assert buffer.to_float32().dtype == np.float32

    def test_exported_samples_are_a_snapshot_copy(self) -> None:
        buffer = PCM16Buffer(60)
        buffer.append(frame_bytes(int16_frame(500)))
        exported = buffer.to_int16()

        buffer.append(frame_bytes(int16_frame(600)))
        assert exported.size == SAMPLES_PER_FRAME

        exported[0] = 0
        assert int(buffer.to_int16()[0]) == 500


class TestErrorHierarchy:
    @pytest.mark.parametrize("error", [PCMFrameSizeError, PCMBufferOverflow])
    def test_pcm_errors_are_value_errors(self, error: type[Exception]) -> None:
        assert issubclass(error, PCMError)
        assert issubclass(error, ValueError)
