import logging
import time

from faster_whisper import WhisperModel


logger = logging.getLogger(__name__)


class WhisperHe:
    LANG = "he"
    MODEL_NAME = "ivrit-ai/whisper-large-v3-turbo-ct2"

    def __init__(self):
        self.model, self._backend = self._create_model()

    @staticmethod
    def _is_cuda_runtime_error(error: Exception) -> bool:
        message = str(error).lower()
        cuda_markers = (
            "cuda",
            "cublas",
            "cudnn",
            "cufft",
            "curand",
            "cudart",
            "nvidia",
            "gpu",
            "dll",
        )
        return isinstance(error, (RuntimeError, OSError)) and any(
            marker in message for marker in cuda_markers
        )

    def _build_model(self, device: str, compute_type: str) -> WhisperModel:
        return WhisperModel(
            self.MODEL_NAME,
            device=device,
            compute_type=compute_type,
            num_workers=1,
        )

    def _create_model(self) -> tuple[WhisperModel, str]:
        try:
            model = self._build_model(device="cuda", compute_type="float16")
            logger.info("WhisperHe initialized on GPU (CUDA).")
            return model, "cuda"
        except Exception as error:
            if not self._is_cuda_runtime_error(error):
                raise
            logger.warning(
                "CUDA is unavailable or misconfigured (%s). Falling back to CPU.",
                error,
            )
            model = self._build_model(device="cpu", compute_type="int8")
            logger.info("WhisperHe initialized on CPU fallback.")
            return model, "cpu"

    def _switch_to_cpu(self) -> None:
        self.model = self._build_model(device="cpu", compute_type="int8")
        self._backend = "cpu"
        logger.info("WhisperHe switched to CPU fallback.")

    def _transcribe_once(self, path: str) -> str:
        segments, _ = self.model.transcribe(
            path,
            temperature=0.0,
            language=self.LANG,
            best_of=1,
            vad_filter=True,
        )
        return " ".join(s.text for s in segments).strip()

    def transcribe(self, path):
        t0 = time.perf_counter()
        try:
            transcript_text = self._transcribe_once(path)
        except Exception as error:
            if self._backend != "cuda" or not self._is_cuda_runtime_error(error):
                raise
            logger.warning(
                "CUDA failed while transcribing (%s). Retrying on CPU fallback.",
                error,
            )
            self._switch_to_cpu()
            transcript_text = self._transcribe_once(path)
        dt = time.perf_counter() - t0
        return transcript_text, f"{dt:.3f} seconds"
