from abc import ABC, abstractmethod
from pathlib import Path


class TranscriptionEngine(ABC):
    @abstractmethod
    def transcribe(self, media_path: Path) -> tuple[str, str]:
        """
        Return transcript text and runtime metadata string.
        """
        raise NotImplementedError
