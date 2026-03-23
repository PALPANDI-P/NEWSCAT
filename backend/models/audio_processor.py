"""
NEWSCAT Audio Processor — Speech-to-text + classification for news audio.
Supports SpeechRecognition (Google Web API) and OpenAI Whisper backends.
Gracefully degrades when optional dependencies are unavailable.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AudioResult:
    """Result of audio processing."""
    success: bool = False
    transcribed_text: str = ""
    confidence: float = 0.0
    error_message: str = ""
    duration_seconds: float = 0.0
    metadata: dict = field(default_factory=dict)


class AudioPreprocessor:
    """Handles audio format conversion and preprocessing."""

    def __init__(self):
        self._ffmpeg_available = self._check_ffmpeg()

    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available on the system."""
        import shutil
        return shutil.which("ffmpeg") is not None

    def is_ffmpeg_available(self) -> bool:
        return self._ffmpeg_available

    def convert_to_wav(self, input_path: str, output_path: str) -> bool:
        """Convert audio file to WAV format using pydub or ffmpeg."""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(input_path)
            audio.export(output_path, format="wav")
            return True
        except ImportError:
            logger.debug("pydub not available, trying ffmpeg directly")
        except Exception as e:
            logger.warning(f"pydub conversion failed: {e}")

        # Fallback: ffmpeg command line
        if self._ffmpeg_available:
            try:
                import subprocess
                result = subprocess.run(
                    ["ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1",
                     "-f", "wav", "-y", output_path],
                    capture_output=True, timeout=60,
                )
                return result.returncode == 0
            except Exception as e:
                logger.warning(f"ffmpeg conversion failed: {e}")

        return False


class AudioProcessor:
    """
    Multi-backend audio processor.
    Priority: Whisper > SpeechRecognition (Google Web API).
    """

    def __init__(self):
        self.preprocessor = AudioPreprocessor()
        self._stt_engine = "none"
        self._whisper_model = None
        self._detect_backends()

    def _detect_backends(self):
        """Detect available speech-to-text backends."""
        # Check Whisper
        try:
            import whisper  # noqa: F401
            self._stt_engine = "whisper"
            logger.info("Whisper detected — using as primary STT engine")
            return
        except ImportError:
            pass

        # Check SpeechRecognition
        try:
            import speech_recognition  # noqa: F401
            self._stt_engine = "speech_recognition"
            logger.info("SpeechRecognition detected — using as STT engine")
            return
        except ImportError:
            pass

        logger.warning("No speech-to-text engine found — audio processing unavailable")

    @property
    def stt_engine(self) -> str:
        return self._stt_engine

    def is_available(self) -> bool:
        return self._stt_engine != "none"

    def get_installation_instructions(self) -> str:
        if self._stt_engine != "none":
            return "All audio dependencies installed"
        return (
            "Install one of:\n"
            "  pip install openai-whisper   # Best accuracy\n"
            "  pip install SpeechRecognition  # Simpler, uses Google API\n"
            "Optional: pip install pydub   # For format conversion\n"
            "Also install ffmpeg for audio preprocessing"
        )

    def process_audio_file(self, file_path: str) -> AudioResult:
        """Transcribe audio file using the best available STT engine."""
        if not os.path.exists(file_path):
            return AudioResult(
                success=False, error_message=f"File not found: {file_path}"
            )

        if not self.is_available():
            return AudioResult(
                success=False,
                error_message="No speech-to-text engine available. " +
                              self.get_installation_instructions(),
            )

        # Convert to WAV if needed
        wav_path = file_path
        needs_cleanup = False
        if not file_path.lower().endswith(".wav"):
            import tempfile
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.close()
            if self.preprocessor.convert_to_wav(file_path, tmp.name):
                wav_path = tmp.name
                needs_cleanup = True
            else:
                # Try processing the original file anyway
                wav_path = file_path

        try:
            if self._stt_engine == "whisper":
                result = self._transcribe_whisper(wav_path)
            elif self._stt_engine == "speech_recognition":
                result = self._transcribe_sr(wav_path)
            else:
                result = AudioResult(success=False, error_message="No STT engine")
                
            # ------------------------------------------------------------------
            # EXPERT AUDIO ACOUSTIC AI (AudioSet Classification)
            # ------------------------------------------------------------------
            if result.success and os.path.exists(wav_path):
                acoustic_tags = []
                try:
                    import torch
                    from transformers import pipeline
                    if not hasattr(self, "_acoustic_pipeline"):
                        # Load ultra-lightweight audio model for CPU 
                        self._acoustic_pipeline = pipeline(
                            "audio-classification", 
                            model="MIT/ast-finetuned-audioset-10-10-0.4593",
                            device=-1 # Force CPU
                        )
                    
                    # Predict top acoustic sounds
                    predictions = self._acoustic_pipeline(wav_path, top_k=3)
                    for p in predictions:
                        if p["score"] > 0.1:
                            labels = p["label"].split(",")
                            acoustic_tags.extend([l.strip().lower() for l in labels])
                    
                    if acoustic_tags:
                        result.metadata["acoustic_scene_tags"] = acoustic_tags
                        scene_string = " ".join(acoustic_tags)
                        result.transcribed_text = f"{result.transcribed_text} [ACOUSTIC_SCENE: {scene_string}]"
                        
                        # Boost confidence since we mapped the acoustic scene
                        result.confidence = min(0.95, result.confidence + 0.3)
                        logger.debug(f"A.I. Acoustic Scene mapped: {scene_string}")
                        
                except ImportError:
                    logger.debug("Transformers/Torch not installed; skipping Expert Acoustic AI.")
                except Exception as e:
                    logger.debug(f"Expert Acoustic AI skipped during processing: {e}")
                    
            return result
                
        finally:
            if needs_cleanup:
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

    def process_audio_data(self, audio_data: bytes, filename: str = "") -> AudioResult:
        """Process audio from raw bytes."""
        import tempfile
        ext = ".wav"
        if filename:
            for audio_ext in [".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac"]:
                if filename.lower().endswith(audio_ext):
                    ext = audio_ext
                    break
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name
            result = self.process_audio_file(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        return result

    # ------------------------------------------------------------------
    # STT Backends
    # ------------------------------------------------------------------

    def _transcribe_whisper(self, wav_path: str) -> AudioResult:
        """Transcribe using OpenAI Whisper."""
        try:
            import whisper
            if self._whisper_model is None:
                self._whisper_model = whisper.load_model("base")
            result = self._whisper_model.transcribe(wav_path, fp16=False)
            text = result.get("text", "").strip()
            # Whisper doesn't return a single confidence score, estimate from segments
            segments = result.get("segments", [])
            if segments:
                avg_conf = sum(
                    s.get("no_speech_prob", 0.0) for s in segments
                ) / len(segments)
                confidence = max(0.1, 1.0 - avg_conf)
            else:
                confidence = 0.8 if text else 0.1

            duration = result.get("segments", [{}])[-1].get("end", 0) if segments else 0

            return AudioResult(
                success=True,
                transcribed_text=text,
                confidence=round(confidence, 3),
                duration_seconds=duration,
                metadata={"engine": "whisper", "language": result.get("language", "en")},
            )
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return AudioResult(success=False, error_message=f"Whisper error: {e}")

    def _transcribe_sr(self, wav_path: str) -> AudioResult:
        """Transcribe using SpeechRecognition (Google Web API)."""
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return AudioResult(
                success=True,
                transcribed_text=text,
                confidence=0.75,  # Google API doesn't return confidence directly
                metadata={"engine": "speech_recognition_google"},
            )
        except Exception as e:
            logger.error(f"SpeechRecognition failed: {e}")
            return AudioResult(success=False, error_message=f"SR error: {e}")


# ------------------------------------------------------------------
# Singleton factory
# ------------------------------------------------------------------

_audio_processor: Optional[AudioProcessor] = None


def get_audio_processor() -> AudioProcessor:
    global _audio_processor
    if _audio_processor is None:
        _audio_processor = AudioProcessor()
    return _audio_processor
