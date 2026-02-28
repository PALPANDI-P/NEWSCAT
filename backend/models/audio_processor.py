"""
NEWSCAT - Audio Processor Module
Handles audio input, speech-to-text conversion using multiple engines
Supports: Whisper, Google Speech Recognition, Sphinx (offline)

Optimized Version:
- Lazy STT engine initialization
- Memory-efficient processing
- Automatic resource cleanup
- Thread-safe operations
- Temporary file management
"""

import os
import io
import logging
import tempfile
import threading
import gc
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import atexit

# Configure logging
logger = logging.getLogger(__name__)

# Global lock for thread-safe STT initialization
_stt_lock = threading.Lock()


@dataclass
class AudioProcessingResult:
    """Result of audio processing"""
    success: bool
    extracted_text: str = ""
    confidence: float = 0.0
    error_message: str = ""
    metadata: Dict[str, Any] = None
    duration: float = 0.0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AudioProcessor:
    """
    Audio processor for news audio content.
    Extracts text using Speech-to-Text (STT) engines.
    
    Features:
    - Lazy STT engine initialization (load on first use)
    - Memory-efficient processing
    - Automatic resource cleanup
    - Thread-safe operations
    """
    
    def __init__(self, preferred_engine: str = 'auto', lazy_init: bool = True):
        self.stt_engine = None
        self.stt_model = None
        self.preferred_engine = preferred_engine
        self._initialized = False
        self._init_lock = threading.Lock()
        
        if not lazy_init:
            self._initialize_stt()
    
    def _ensure_initialized(self):
        """Ensure STT is initialized (lazy loading)"""
        if self._initialized:
            return
        
        with self._init_lock:
            if not self._initialized:
                self._initialize_stt()
                self._initialized = True
    
    def _initialize_stt(self):
        """Initialize Speech-to-Text engine with fallback options"""
        with _stt_lock:
            if self.stt_engine is not None:
                return
            
            engines_to_try = []
            
            if self.preferred_engine == 'auto':
                # Try engines in order of accuracy
                engines_to_try = ['whisper', 'google', 'sphinx']
            else:
                engines_to_try = [self.preferred_engine]
            
            for engine in engines_to_try:
                try:
                    if engine == 'whisper':
                        self._init_whisper()
                        return
                    elif engine == 'google':
                        self._init_google()
                        return
                    elif engine == 'sphinx':
                        self._init_sphinx()
                        return
                except Exception as e:
                    logger.debug(f"{engine} not available: {e}")
                    continue
            
            # No STT available
            self.stt_engine = 'none'
            logger.warning("No Speech-to-Text engine available. Audio classification will return helpful error message.")
    
    def _init_whisper(self):
        """Initialize OpenAI Whisper (most accurate, runs locally)"""
        try:
            import whisper
            
            # Use base model for balance of speed/accuracy
            self.stt_model = whisper.load_model("base")
            self.stt_engine = 'whisper'
            logger.info("Whisper STT initialized successfully")
        except ImportError:
            logger.error("Whisper not installed. Install with: pip install openai-whisper")
            raise RuntimeError("Whisper not installed. Install with: pip install openai-whisper")
    
    def _init_google(self):
        """Initialize Google Speech Recognition (requires internet)"""
        try:
            import speech_recognition as sr
            
            self.stt_model = sr.Recognizer()
            self.stt_model.energy_threshold = 300
            self.stt_model.dynamic_energy_threshold = True
            self.stt_engine = 'google'
            logger.info("Google Speech Recognition initialized successfully")
        except ImportError:
            logger.error("SpeechRecognition not installed. Install with: pip install SpeechRecognition")
            raise RuntimeError("SpeechRecognition not installed. Install with: pip install SpeechRecognition")
    
    def _init_sphinx(self):
        """Initialize CMU Sphinx (offline, less accurate)"""
        try:
            import speech_recognition as sr
            
            self.stt_model = sr.Recognizer()
            self.stt_engine = 'sphinx'
            logger.info("CMU Sphinx STT initialized successfully")
        except ImportError:
            logger.error("SpeechRecognition not installed. Install with: pip install SpeechRecognition")
            raise RuntimeError("SpeechRecognition not installed. Install with: pip install SpeechRecognition")
    
    def is_available(self) -> bool:
        """Check if audio processing is available"""
        self._ensure_initialized()
        return self.stt_engine is not None and self.stt_engine != 'none'
    
    def get_installation_instructions(self) -> str:
        """Get installation instructions for STT"""
        return """
To enable audio classification, install one of the following:

Option 1 - OpenAI Whisper (Recommended, Most Accurate):
  pip install openai-whisper
  (First run will download model ~150MB)

Option 2 - Google Speech Recognition (Requires Internet):
  pip install SpeechRecognition pyaudio
  
Option 3 - CMU Sphinx (Offline, Less Accurate):
  pip install SpeechRecognition pocketsphinx

For audio file format support:
  pip install pydub ffmpeg-python
  (Also requires FFmpeg installed on system)

After installation, restart the NEWSCAT server.
"""
    
    def process_audio_file(self, audio_path: str) -> AudioProcessingResult:
        """Process an audio file and extract text"""
        self._ensure_initialized()
        
        temp_file = None
        
        try:
            # Get audio duration
            duration = self._get_audio_duration(audio_path)
            
            # Check duration (max 5 minutes for memory efficiency)
            max_duration = 300  # 5 minutes
            if duration > max_duration:
                return AudioProcessingResult(
                    success=False,
                    error_message=f"Audio too long. Maximum duration is {max_duration} seconds.",
                    duration=duration
                )
            
            if self.stt_engine == 'whisper':
                return self._process_with_whisper(audio_path, duration)
            elif self.stt_engine in ['google', 'sphinx']:
                return self._process_with_sr(audio_path, duration)
            else:
                return AudioProcessingResult(
                    success=False,
                    error_message=f"No STT engine available.{self.get_installation_instructions()}",
                    duration=duration
                )
                
        except Exception as e:
            logger.error(f"Error processing audio file: {e}")
            return AudioProcessingResult(
                success=False,
                error_message=f"Failed to process audio: {str(e)}"
            )
    
    def process_audio_bytes(self, audio_bytes: bytes, file_extension: str = '.wav') -> AudioProcessingResult:
        """Process audio bytes and extract text"""
        self._ensure_initialized()
        
        temp_file = None
        
        try:
            # Save bytes to temporary file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=file_extension,
                prefix='newscat_audio_'
            )
            temp_file.write(audio_bytes)
            temp_file.close()
            
            result = self.process_audio_file(temp_file.name)
            return result
            
        except Exception as e:
            logger.error(f"Error processing audio bytes: {e}")
            return AudioProcessingResult(
                success=False,
                error_message=f"Failed to process audio: {str(e)}"
            )
        finally:
            # Cleanup temp file
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.remove(temp_file.name)
                except Exception:
                    pass
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds"""
        try:
            # Try with pydub first
            from pydub import AudioSegment
            audio = AudioSegment.from_file(audio_path)
            return len(audio) / 1000.0  # Convert ms to seconds
        except Exception:
            try:
                # Fallback to wave module for WAV files
                import wave
                with wave.open(audio_path, 'r') as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    return frames / float(rate)
            except Exception:
                # Unknown duration
                return 0.0
    
    def _process_with_whisper(self, audio_path: str, duration: float) -> AudioProcessingResult:
        """Process audio with Whisper"""
        try:
            # Transcribe
            result = self.stt_model.transcribe(audio_path, language='en')
            extracted_text = result.get('text', '').strip()
            
            # Calculate confidence (Whisper doesn't provide per-word confidence)
            confidence = 0.85  # Default confidence for Whisper
            
            metadata = {
                'engine': 'whisper',
                'language': result.get('language', 'en'),
                'duration': duration
            }
            
            return AudioProcessingResult(
                success=True,
                extracted_text=extracted_text,
                confidence=confidence,
                metadata=metadata,
                duration=duration
            )
            
        except Exception as e:
            logger.error(f"Whisper processing error: {e}")
            return AudioProcessingResult(
                success=False,
                error_message=f"Whisper transcription failed: {str(e)}",
                duration=duration
            )
    
    def _process_with_sr(self, audio_path: str, duration: float) -> AudioProcessingResult:
        """Process audio with SpeechRecognition (Google or Sphinx)"""
        try:
            import speech_recognition as sr
            
            # Convert to WAV if needed
            wav_path = audio_path
            temp_wav = None
            
            if not audio_path.lower().endswith('.wav'):
                try:
                    from pydub import AudioSegment
                    audio = AudioSegment.from_file(audio_path)
                    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    audio.export(temp_wav.name, format='wav')
                    wav_path = temp_wav.name
                except Exception as e:
                    return AudioProcessingResult(
                        success=False,
                        error_message=f"Failed to convert audio to WAV: {str(e)}",
                        duration=duration
                    )
            
            # Transcribe
            with sr.AudioFile(wav_path) as source:
                # Noise reduction
                self.stt_model.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = self.stt_model.record(source)
                
                if self.stt_engine == 'google':
                    text = self.stt_model.recognize_google(audio_data)
                else:  # sphinx
                    text = self.stt_model.recognize_sphinx(audio_data)
                
                confidence = 0.75 if self.stt_engine == 'google' else 0.60
            
            # Cleanup temp WAV
            if temp_wav and os.path.exists(temp_wav.name):
                try:
                    os.remove(temp_wav.name)
                except Exception:
                    pass
            
            metadata = {
                'engine': self.stt_engine,
                'duration': duration
            }
            
            return AudioProcessingResult(
                success=True,
                extracted_text=text,
                confidence=confidence,
                metadata=metadata,
                duration=duration
            )
            
        except sr.UnknownValueError:
            return AudioProcessingResult(
                success=False,
                error_message="Could not understand audio. Please try clearer audio.",
                duration=duration
            )
        except sr.RequestError as e:
            return AudioProcessingResult(
                success=False,
                error_message=f"Speech recognition service error: {str(e)}",
                duration=duration
            )
        except Exception as e:
            logger.error(f"SpeechRecognition processing error: {e}")
            return AudioProcessingResult(
                success=False,
                error_message=f"Transcription failed: {str(e)}",
                duration=duration
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get audio processor status"""
        self._ensure_initialized()
        
        return {
            'available': self.is_available(),
            'engine': self.stt_engine if self.stt_engine != 'none' else None,
            'supported_formats': ['wav', 'mp3', 'm4a', 'flac', 'ogg', 'webm'] if self.is_available() else [],
            'installation_instructions': self.get_installation_instructions() if not self.is_available() else None
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if self.stt_model is not None:
            # Whisper cleanup
            if self.stt_engine == 'whisper':
                del self.stt_model
            self.stt_model = None
        
        self.stt_engine = None
        self._initialized = False
        gc.collect()
        logger.info("Audio processor resources cleaned up")


# Singleton instance with lazy initialization
_audio_processor: Optional[AudioProcessor] = None
_processor_lock = threading.Lock()


def get_audio_processor() -> AudioProcessor:
    """Get or create audio processor instance (lazy singleton)"""
    global _audio_processor
    
    if _audio_processor is None:
        with _processor_lock:
            if _audio_processor is None:
                _audio_processor = AudioProcessor(lazy_init=True)
    
    return _audio_processor


def cleanup_audio_processor():
    """Cleanup audio processor resources"""
    global _audio_processor
    
    if _audio_processor is not None:
        _audio_processor.cleanup()
        _audio_processor = None


# Register cleanup on exit
atexit.register(cleanup_audio_processor)
