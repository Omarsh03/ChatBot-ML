import pytest

from app.transcription.engines import whisper_he


def test_whisper_he_prefers_cuda_when_available() -> None:
    calls: list[tuple[str, str, str, int]] = []

    def fake_whisper_model(
        model_name: str, *, device: str, compute_type: str, num_workers: int
    ) -> object:
        calls.append((model_name, device, compute_type, num_workers))
        return object()

    monkey = pytest.MonkeyPatch()
    monkey.setattr(whisper_he, "WhisperModel", fake_whisper_model)
    try:
        whisper_he.WhisperHe()
    finally:
        monkey.undo()

    assert len(calls) == 1
    assert calls[0][1] == "cuda"
    assert calls[0][2] == "float16"


def test_whisper_he_falls_back_to_cpu_if_cuda_runtime_missing() -> None:
    calls: list[tuple[str, str, str, int]] = []

    def fake_whisper_model(
        model_name: str, *, device: str, compute_type: str, num_workers: int
    ) -> object:
        calls.append((model_name, device, compute_type, num_workers))
        if device == "cuda":
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
        return {"device": device, "compute_type": compute_type}

    monkey = pytest.MonkeyPatch()
    monkey.setattr(whisper_he, "WhisperModel", fake_whisper_model)
    try:
        instance = whisper_he.WhisperHe()
    finally:
        monkey.undo()

    assert len(calls) == 2
    assert calls[0][1] == "cuda"
    assert calls[1][1] == "cpu"
    assert calls[1][2] == "int8"
    assert instance.model["device"] == "cpu"


def test_whisper_he_reraises_non_cuda_initialization_errors() -> None:
    def fake_whisper_model(
        model_name: str, *, device: str, compute_type: str, num_workers: int
    ) -> object:
        raise RuntimeError("Unsupported model configuration")

    monkey = pytest.MonkeyPatch()
    monkey.setattr(whisper_he, "WhisperModel", fake_whisper_model)
    try:
        with pytest.raises(RuntimeError, match="Unsupported model configuration"):
            whisper_he.WhisperHe()
    finally:
        monkey.undo()


def test_whisper_he_falls_back_to_cpu_when_cuda_fails_during_transcribe() -> None:
    calls: list[tuple[str, str, str, int]] = []

    class _Segment:
        def __init__(self, text: str):
            self.text = text

    class _Model:
        def __init__(self, device: str):
            self._device = device

        def transcribe(self, path: str, **kwargs):
            def _segments():
                if self._device == "cuda":
                    raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
                yield _Segment("hello")
                yield _Segment("world")

            return _segments(), None

    def fake_whisper_model(
        model_name: str, *, device: str, compute_type: str, num_workers: int
    ) -> object:
        calls.append((model_name, device, compute_type, num_workers))
        return _Model(device=device)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(whisper_he, "WhisperModel", fake_whisper_model)
    try:
        instance = whisper_he.WhisperHe()
        text, runtime = instance.transcribe("lecture.mp3")
    finally:
        monkey.undo()

    assert len(calls) == 2
    assert calls[0][1] == "cuda"
    assert calls[1][1] == "cpu"
    assert instance._backend == "cpu"
    assert text == "hello world"
    assert runtime.endswith("seconds")
