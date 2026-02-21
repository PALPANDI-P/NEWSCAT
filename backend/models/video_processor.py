

"""
NEWSCAT - Video Processor Module
Handles video input, frame extraction, and OCR for text extraction
Supports: MP4, AVI, MOV, WebM, MKV
"""

import os
import io
import logging
import tempfile
import base64
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import math

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class VideoProcessingResult:
    """Result of video processing"""
    success: bool
    extracted_text: str = ""
    confidence: float = 0.0
    error_message: str = ""
    metadata: Dict[str, Any] = None
    frames_processed: int = 0
    duration: float = 0.0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class VideoProcessor:
    """
    Video processor for news video content.
    Extracts frames and uses OCR to extract text.
    Can also extract audio for speech-to-text.
    """
    
    def __init__(self, lazy_init: bool = True):
        self.ocr_processor = None
        self.audio_processor = None
        self._initialized = False
        
        if not lazy_init:
            self._initialize_processors()
    
    def _ensure_initialized(self):
        """Ensure processors are initialized (lazy loading)"""
        if self._initialized:
            return
        self._initialize_processors()
        self._initialized = True
    
    def _initialize_processors(self):
        """Initialize OCR and audio processors"""
        try:
            from backend.models.image_processor import get_image_processor
            self.ocr_processor = get_image_processor()
        except Exception as e:
            logger.debug(f"OCR processor not available: {e}")
        
        try:
            from backend.models.audio_processor import get_audio_processor
            self.audio_processor = get_audio_processor()
        except Exception as e:
            logger.debug(f"Audio processor not available: {e}")
    
    def is_available(self) -> bool:
        """Check if video processing is available"""
        try:
            import cv2
            return True
        except ImportError:
            return False
    
    def get_installation_instructions(self) -> str:
        """Get installation instructions for video processing"""
        return """
To enable video classification, install the following:

Required:
  pip install opencv-python-headless

For audio extraction from video:
  pip install moviepy pydub ffmpeg-python

For OCR (text from video frames):
  pip install easyocr

For Speech-to-Text (audio from video):
  pip install openai-whisper

Note: FFmpeg must be installed on your system for video processing.
  - Windows: Download from https://ffmpeg.org and add to PATH
  - Mac: brew install ffmpeg
  - Linux: sudo apt install ffmpeg

After installation, restart the NEWSCAT server.
"""
    
    def process_video_file(self, video_path: str, extract_audio: bool = True) -> VideoProcessingResult:
        """Process a video file and extract text from frames and audio"""
        # Ensure processors are initialized
        self._ensure_initialized()
        
        try:
            import cv2
            
            # Get video info
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return VideoProcessingResult(
                    success=False,
                    error_message="Could not open video file"
                )
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            logger.info(f"Processing video: {frame_count} frames, {duration:.1f}s, {fps:.1f} fps")
            
            # Extract frames at intervals
            extracted_texts = []
            frames_processed = 0
            
            # Process frames (every 2 seconds for efficiency)
            frame_interval = int(fps * 2) if fps > 0 else 30
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Only process at intervals
                if frames_processed % frame_interval == 0:
                    text = self._extract_text_from_frame(frame)
                    if text and len(text.strip()) > 10:  # Minimum text threshold
                        extracted_texts.append(text)
                
                frames_processed += 1
            
            cap.release()
            
            # Combine extracted texts
            ocr_text = self._combine_texts(extracted_texts)
            
            # Extract audio if enabled
            audio_text = ""
            if extract_audio and self.audio_processor and self.audio_processor.is_available():
                audio_result = self._extract_audio_from_video(video_path)
                if audio_result:
                    audio_text = audio_result
            
            # Combine OCR and audio text
            final_text = self._combine_texts([ocr_text, audio_text])
            
            if not final_text or len(final_text.strip()) < 20:
                return VideoProcessingResult(
                    success=False,
                    error_message="Could not extract enough text from video. Please try a video with clearer text or speech.",
                    frames_processed=frames_processed,
                    duration=duration
                )
            
            metadata = {
                'fps': fps,
                'total_frames': frame_count,
                'frames_processed': frames_processed,
                'duration': duration,
                'ocr_text_length': len(ocr_text),
                'audio_text_length': len(audio_text),
                'text_length': len(final_text),
                'word_count': len(final_text.split())
            }
            
            return VideoProcessingResult(
                success=True,
                extracted_text=final_text,
                confidence=0.75,  # Default confidence for video
                metadata=metadata,
                frames_processed=frames_processed,
                duration=duration
            )
            
        except ImportError:
            return VideoProcessingResult(
                success=False,
                error_message=f"OpenCV not installed.{self.get_installation_instructions()}"
            )
        except Exception as e:
            logger.error(f"Error processing video file: {e}")
            return VideoProcessingResult(
                success=False,
                error_message=f"Failed to process video: {str(e)}"
            )
    
    def process_video_bytes(self, video_bytes: bytes, file_extension: str = '.mp4') -> VideoProcessingResult:
        """Process video bytes and extract text"""
        try:
            # Save bytes to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
                tmp.write(video_bytes)
                tmp_path = tmp.name
            
            result = self.process_video_file(tmp_path)
            
            # Cleanup
            try:
                os.unlink(tmp_path)
            except:
                pass
            
            return result
        except Exception as e:
            logger.error(f"Error processing video bytes: {e}")
            return VideoProcessingResult(
                success=False,
                error_message=f"Failed to process video: {str(e)}"
            )
    
    def _extract_text_from_frame(self, frame) -> str:
        """Extract text from a video frame using OCR"""
        if not self.ocr_processor or not self.ocr_processor.is_available():
            return ""
        
        try:
            from PIL import Image
            import numpy as np
            
            # Convert OpenCV frame (BGR) to PIL Image (RGB)
            frame_rgb = frame[:, :, ::-1]  # BGR to RGB
            image = Image.fromarray(frame_rgb)
            
            # Use OCR processor
            result = self.ocr_processor._extract_text_from_image(image)
            if result.success:
                return result.extracted_text
            return ""
        except Exception as e:
            logger.debug(f"Frame OCR error: {e}")
            return ""
    
    def _extract_audio_from_video(self, video_path: str) -> Optional[str]:
        """Extract audio from video and transcribe"""
        try:
            from moviepy.editor import VideoFileClip
            import tempfile
            
            # Extract audio
            video = VideoFileClip(video_path)
            if video.audio is None:
                return None
            
            # Save audio to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                audio_path = tmp.name
            
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            video.close()
            
            # Process audio
            result = self.audio_processor.process_audio_file(audio_path)
            
            # Cleanup
            try:
                os.unlink(audio_path)
            except:
                pass
            
            if result.success:
                return result.extracted_text
            return None
        except Exception as e:
            logger.debug(f"Audio extraction error: {e}")
            return None
    
    def _combine_texts(self, texts: List[str]) -> str:
        """Combine multiple extracted texts, removing duplicates"""
        if not texts:
            return ""
        
        # Combine all texts
        combined = " ".join(t.strip() for t in texts if t and t.strip())
        
        # Remove duplicate sentences
        sentences = combined.split('. ')
        unique_sentences = []
        seen = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in seen:
                # Check for similar sentences (fuzzy match)
                is_duplicate = False
                for seen_sentence in seen:
                    if self._similar(sentence, seen_sentence, threshold=0.8):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_sentences.append(sentence)
                    seen.add(sentence)
        
        return '. '.join(unique_sentences)
    
    def _similar(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Check if two texts are similar"""
        if not text1 or not text2:
            return False
        
        # Simple similarity check using word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return (intersection / union) >= threshold if union > 0 else False
    
    def get_status(self) -> Dict[str, Any]:
        """Get video processor status"""
        return {
            'available': self.is_available(),
            'ocr_available': self.ocr_processor.is_available() if self.ocr_processor else False,
            'audio_available': self.audio_processor.is_available() if self.audio_processor else False,
            'supported_formats': ['mp4', 'avi', 'mov', 'webm', 'mkv'] if self.is_available() else [],
            'installation_instructions': self.get_installation_instructions() if not self.is_available() else None
        }


# Singleton instance
_video_processor = None

def get_video_processor() -> VideoProcessor:
    """Get or create video processor instance"""
    global _video_processor
    if _video_processor is None:
        _video_processor = VideoProcessor()
    return _video_processor


# Convenience functions
def process_video(video_input, input_type: str = 'file') -> VideoProcessingResult:
    """
    Process video from various input types.
    
    Args:
        video_input: Video data (file path or bytes)
        input_type: Type of input ('file', 'bytes')
    
    Returns:
        VideoProcessingResult with extracted text
    """
    processor = get_video_processor()
    
    if input_type == 'file':
        return processor.process_video_file(video_input)
    elif input_type == 'bytes':
        return processor.process_video_bytes(video_input)
    else:
        return VideoProcessingResult(
            success=False,
            error_message=f"Unknown input type: {input_type}"
        )


if __name__ == "__main__":
    # Test the video processor
    processor = get_video_processor()
    print(f"Video Processor Status: {processor.get_status()}")
