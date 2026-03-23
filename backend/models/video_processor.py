"""
NEWSCAT Video Processor — Combined audio + visual analysis for news video classification.
Extracts audio track for STT and key frames for OCR, merges results.
Supports OpenCV and MoviePy backends with graceful degradation.
"""

import os
import logging
import tempfile
from dataclasses import dataclass, field
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class VideoResult:
    """Result of video processing."""
    success: bool = False
    extracted_text: str = ""
    transcribed_audio: str = ""
    confidence: float = 0.0
    error_message: str = ""
    duration_seconds: float = 0.0
    frame_count: int = 0
    metadata: dict = field(default_factory=dict)


class VideoProcessor:
    """
    Multi-modal video processor.
    1. Extracts audio → routes to AudioProcessor for STT.
    2. Extracts key frames → routes to ImageProcessor for OCR.
    3. Merges text from both for classification.
    """

    def __init__(self):
        self._cv2_available = False
        self._moviepy_available = False
        self._detect_backends()

    def _detect_backends(self):
        """Detect available video processing backends."""
        try:
            import cv2  # noqa: F401
            self._cv2_available = True
            logger.info("OpenCV detected — video frame extraction available")
        except ImportError:
            logger.debug("OpenCV not installed")

        try:
            from moviepy.editor import VideoFileClip  # noqa: F401
            self._moviepy_available = True
            logger.info("MoviePy detected — video audio extraction available")
        except ImportError:
            logger.debug("MoviePy not installed")

    def is_available(self) -> bool:
        """At least one backend must be available."""
        return self._cv2_available or self._moviepy_available

    def get_installation_instructions(self) -> str:
        parts = []
        if not self._cv2_available:
            parts.append("pip install opencv-python-headless")
        if not self._moviepy_available:
            parts.append("pip install moviepy")
        return " && ".join(parts) if parts else "All video dependencies installed"

    def get_dependencies_status(self) -> Dict[str, bool]:
        return {
            "opencv": self._cv2_available,
            "moviepy": self._moviepy_available,
        }

    def process_video_file(self, file_path: str) -> VideoResult:
        """
        Process a video file: extract audio + key frames, merge text results.
        """
        if not os.path.exists(file_path):
            return VideoResult(
                success=False, error_message=f"File not found: {file_path}"
            )

        if not self.is_available():
            return VideoResult(
                success=False,
                error_message="No video processing backend. " +
                              self.get_installation_instructions(),
            )

        all_text_parts: List[str] = []
        transcribed_audio = ""
        frame_texts: List[str] = []
        duration = 0.0
        frame_count = 0

        # 1. Extract and process audio track
        audio_text, audio_conf, duration = self._extract_and_process_audio(file_path)
        if audio_text:
            transcribed_audio = audio_text
            all_text_parts.append(audio_text)

        # 2. Extract and process key frames
        frame_texts, frame_count = self._extract_and_process_frames(file_path)
        all_text_parts.extend(frame_texts)

        # Signal Weighting: Repeat the audio transcription 3 times to ensure 
        # it dominates the classification logic over potentially noisy OCR frames.
        weighted_parts = []
        if transcribed_audio:
            # Injecting with priority tag and repetition
            weighted_parts.append(f"[PRIORITY_AUDIO]: {transcribed_audio} {transcribed_audio} {transcribed_audio}")
        
        # Add frame texts as supporting context
        weighted_parts.extend(frame_texts)
        
        combined_text = " ".join(weighted_parts).strip()

        # Calculate overall confidence
        if transcribed_audio and frame_texts:
            confidence = min(0.95, (audio_conf + 0.5) / 2 + 0.2)
        elif transcribed_audio:
            confidence = audio_conf
        elif frame_texts:
            confidence = 0.5
        else:
            confidence = 0.2

        return VideoResult(
            success=True,
            extracted_text=combined_text,
            transcribed_audio=transcribed_audio,
            confidence=round(confidence, 3),
            duration_seconds=duration,
            frame_count=frame_count,
            metadata={
                "has_audio": bool(transcribed_audio),
                "has_frames": bool(frame_texts),
                "cv2_available": self._cv2_available,
                "moviepy_available": self._moviepy_available,
            },
        )

    def process_video_data(self, video_data: bytes, filename: str = "") -> VideoResult:
        """Process video from raw bytes."""
        ext = ".mp4"
        if filename:
            for vid_ext in [".avi", ".mkv", ".mov", ".webm", ".flv"]:
                if filename.lower().endswith(vid_ext):
                    ext = vid_ext
                    break
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(video_data)
                tmp_path = tmp.name
            result = self.process_video_file(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        return result

    # ------------------------------------------------------------------
    # Audio extraction
    # ------------------------------------------------------------------

    def _extract_and_process_audio(self, video_path: str) -> tuple:
        """Extract audio from video and transcribe."""
        audio_text = ""
        confidence = 0.0
        duration = 0.0

        if self._moviepy_available:
            try:
                from moviepy.editor import VideoFileClip
                clip = VideoFileClip(video_path)
                duration = clip.duration or 0.0

                if clip.audio is not None:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".wav"
                    ) as tmp:
                        audio_path = tmp.name
                    clip.audio.write_audiofile(
                        audio_path, fps=16000, nbytes=2,
                        codec="pcm_s16le", logger=None,
                    )
                    clip.close()

                    # Route to audio processor
                    from backend.models.audio_processor import get_audio_processor
                    audio_proc = get_audio_processor()
                    if audio_proc.is_available():
                        result = audio_proc.process_audio_file(audio_path)
                        if result.success:
                            audio_text = result.transcribed_text
                            confidence = result.confidence

                    try:
                        os.remove(audio_path)
                    except Exception:
                        pass
                else:
                    clip.close()
            except Exception as e:
                logger.warning(f"Audio extraction failed: {e}")

        return audio_text, confidence, duration

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------

    def _extract_and_process_frames(self, video_path: str) -> tuple:
        """Extract key frames and run OCR on them."""
        frame_texts: List[str] = []
        frame_count = 0

        if not self._cv2_available:
            return frame_texts, frame_count

        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return frame_texts, frame_count

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30

            # Sample up to 5 frames evenly spread across the video
            sample_count = min(5, max(1, total_frames // int(fps * 5)))
            interval = max(1, total_frames // (sample_count + 1))

            from backend.models.image_processor import get_image_processor
            img_proc = get_image_processor()

            for i in range(1, sample_count + 1):
                frame_num = i * interval
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    continue

                frame_count += 1

                if img_proc.is_available():
                    # Save frame to temp file and process
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".jpg"
                    ) as tmp:
                        frame_path = tmp.name
                    cv2.imwrite(frame_path, frame)

                    result = img_proc.process_image_file(frame_path)
                    if result.success and result.extracted_text.strip():
                        frame_texts.append(result.extracted_text)

                    try:
                        os.remove(frame_path)
                    except Exception:
                        pass

            cap.release()
        except Exception as e:
            logger.warning(f"Frame extraction failed: {e}")

        return frame_texts, frame_count


# ------------------------------------------------------------------
# Singleton factory
# ------------------------------------------------------------------

_video_processor: Optional[VideoProcessor] = None


def get_video_processor() -> VideoProcessor:
    global _video_processor
    if _video_processor is None:
        _video_processor = VideoProcessor()
    return _video_processor
