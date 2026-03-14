from pathlib import Path

from app.transcription.engines.base import TranscriptionEngine
from app.transcription.engines.whisper_he import WhisperHe


class WhisperHeEngine(TranscriptionEngine):
    def __init__(self) -> None:
        self._engine = WhisperHe()

    def transcribe(self, media_path: Path) -> tuple[str, str]:
        return self._engine.transcribe(str(media_path))
