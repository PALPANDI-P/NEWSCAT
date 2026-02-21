"""
NEWSCAT - Image Processor Module
Handles image input, OCR text extraction, and image classification
With graceful fallback when OCR is not available

Optimized Version:
- Lazy OCR engine initialization
- Memory-efficient processing
- Automatic resource cleanup
- Thread-safe operations
"""

import os
import io
import base64
import logging
import threading
import gc
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import requests
from pathlib import Path
import tempfile
import atexit

# Configure logging
logger = logging.getLogger(__name__)

# Global lock for thread-safe OCR initialization
_ocr_lock = threading.Lock()


@dataclass
class ImageProcessingResult:
    """Result of image processing"""
    success: bool
    extracted_text: str = ""
    confidence: float = 0.0
    error_message: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ImageProcessor:
    """
    Image processor for news article images.
    Extracts text using OCR and prepares content for classification.
    
    Features:
    - Lazy OCR engine initialization (load on first use)
    - Memory-efficient processing
    - Automatic resource cleanup
    - Thread-safe operations
    """
    
    def __init__(self, lazy_init: bool = True):
        self.ocr_engine = None
        self.ocr_reader = None
        self._initialized = False
        self._init_lock = threading.Lock()
        
        if not lazy_init:
            self._initialize_ocr()
    
    def _ensure_initialized(self):
        """Ensure OCR is initialized (lazy loading)"""
        if self._initialized:
            return
        
        with self._init_lock:
            if not self._initialized:
                self._initialize_ocr()
                self._initialized = True
    
    def _initialize_ocr(self):
        """Initialize OCR engine with fallback options"""
        with _ocr_lock:
            if self.ocr_engine is not None:
                return
            
            # Try EasyOCR first (pure Python, no external binary needed)
            try:
                import easyocr
                self.ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                self.ocr_engine = 'easyocr'
                logger.info("EasyOCR initialized successfully")
                return
            except Exception as e:
                logger.debug(f"EasyOCR not available: {e}")
            
            # Try Tesseract as fallback (requires binary installation)
            try:
                import pytesseract
                # Check if tesseract is available
                pytesseract.get_tesseract_version()
                self.ocr_engine = 'tesseract'
                logger.info("Tesseract OCR initialized successfully")
                return
            except Exception as e:
                logger.debug(f"Tesseract not available: {e}")
            
            # No OCR available - but we'll handle this gracefully
            self.ocr_engine = 'none'
            logger.warning("No OCR engine available. Image classification will return helpful error message.")
    
    def is_available(self) -> bool:
        """Check if image processing is available"""
        self._ensure_initialized()
        return self.ocr_engine is not None and self.ocr_engine != 'none'
    
    def get_installation_instructions(self) -> str:
        """Get installation instructions for OCR"""
        return """
To enable image classification, install one of the following:

Option 1 - EasyOCR (Recommended, Pure Python):
  pip install easyocr

Option 2 - Tesseract (Requires binary installation):
  1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
  2. Install to default location
  3. pip install pytesseract

After installation, restart the NEWSCAT server.
"""
    
    def process_image_file(self, image_path: str) -> ImageProcessingResult:
        """Process an image file and extract text"""
        self._ensure_initialized()
        
        try:
            from PIL import Image
            
            # Open and optimize image
            with Image.open(image_path) as image:
                # Convert to RGB if necessary
                if image.mode not in ('RGB', 'L'):
                    image = image.convert('RGB')
                
                # Resize large images for memory efficiency
                max_dimension = 2000
                if max(image.size) > max_dimension:
                    ratio = max_dimension / max(image.size)
                    new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                    logger.debug(f"Resized image from {image.size} to {new_size}")
                
                return self._extract_text_from_image(image)
                
        except Exception as e:
            logger.error(f"Error processing image file: {e}")
            return ImageProcessingResult(
                success=False,
                error_message=f"Failed to process image: {str(e)}"
            )
    
    def process_image_bytes(self, image_bytes: bytes) -> ImageProcessingResult:
        """Process image bytes and extract text"""
        self._ensure_initialized()
        
        try:
            from PIL import Image
            
            # Use BytesIO for memory efficiency
            with io.BytesIO(image_bytes) as buffer:
                with Image.open(buffer) as image:
                    # Convert to RGB if necessary
                    if image.mode not in ('RGB', 'L'):
                        image = image.convert('RGB')
                    
                    return self._extract_text_from_image(image)
                    
        except Exception as e:
            logger.error(f"Error processing image bytes: {e}")
            return ImageProcessingResult(
                success=False,
                error_message=f"Failed to process image: {str(e)}"
            )
    
    def process_image_url(self, url: str) -> ImageProcessingResult:
        """Download and process image from URL"""
        self._ensure_initialized()
        
        try:
            # Validate URL
            if not url.startswith(('http://', 'https://')):
                return ImageProcessingResult(
                    success=False,
                    error_message="Invalid URL format"
                )
            
            # Download image with timeout
            response = requests.get(
                url, 
                timeout=15,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                },
                stream=True  # Stream for memory efficiency
            )
            response.raise_for_status()
            
            # Check content length
            content_length = int(response.headers.get('content-length', 0))
            max_size = 10 * 1024 * 1024  # 10MB max
            
            if content_length > max_size:
                return ImageProcessingResult(
                    success=False,
                    error_message="Image too large. Maximum size is 10MB."
                )
            
            # Process image
            return self.process_image_bytes(response.content)
            
        except requests.Timeout:
            logger.error(f"Timeout downloading image from URL")
            return ImageProcessingResult(
                success=False,
                error_message="Timeout downloading image. Please try a different URL."
            )
        except requests.RequestException as e:
            logger.error(f"Error downloading image from URL: {e}")
            return ImageProcessingResult(
                success=False,
                error_message=f"Failed to download image: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error processing image URL: {e}")
            return ImageProcessingResult(
                success=False,
                error_message=f"Failed to process image: {str(e)}"
            )
    
    def process_base64_image(self, base64_data: str) -> ImageProcessingResult:
        """Process base64 encoded image"""
        self._ensure_initialized()
        
        try:
            # Remove data URL prefix if present
            if ',' in base64_data:
                base64_data = base64_data.split(',')[1]
            
            image_bytes = base64.b64decode(base64_data)
            return self.process_image_bytes(image_bytes)
            
        except Exception as e:
            logger.error(f"Error processing base64 image: {e}")
            return ImageProcessingResult(
                success=False,
                error_message=f"Failed to decode image: {str(e)}"
            )
    
    def _extract_text_from_image(self, image) -> ImageProcessingResult:
        """Extract text from PIL Image using available OCR engine"""
        if not self.ocr_engine or self.ocr_engine == 'none':
            return ImageProcessingResult(
                success=False,
                error_message=f"No OCR engine available.{self.get_installation_instructions()}"
            )
        
        try:
            extracted_text = ""
            confidence = 0.0
            
            if self.ocr_engine == 'easyocr':
                extracted_text, confidence = self._extract_with_easyocr(image)
            elif self.ocr_engine == 'tesseract':
                extracted_text, confidence = self._extract_with_tesseract(image)
            
            # Clean up extracted text
            extracted_text = self._clean_text(extracted_text)
            
            # Calculate metadata
            metadata = {
                'ocr_engine': self.ocr_engine,
                'image_size': image.size if hasattr(image, 'size') else None,
                'text_length': len(extracted_text),
                'word_count': len(extracted_text.split()) if extracted_text else 0
            }
            
            return ImageProcessingResult(
                success=True,
                extracted_text=extracted_text,
                confidence=confidence,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return ImageProcessingResult(
                success=False,
                error_message=f"OCR failed: {str(e)}"
            )
    
    def _extract_with_tesseract(self, image) -> Tuple[str, float]:
        """Extract text using Tesseract OCR"""
        import pytesseract
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get OCR data with confidence
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        # Extract text and calculate average confidence
        texts = []
        confidences = []
        
        for i, text in enumerate(data['text']):
            if text.strip():
                texts.append(text)
                conf = data['conf'][i]
                if conf > 0:  # Valid confidence
                    confidences.append(conf)
        
        extracted_text = ' '.join(texts)
        avg_confidence = sum(confidences) / len(confidences) / 100.0 if confidences else 0.5
        
        return extracted_text, avg_confidence
    
    def _extract_with_easyocr(self, image) -> Tuple[str, float]:
        """Extract text using EasyOCR"""
        import numpy as np
        
        # Convert PIL Image to numpy array
        image_array = np.array(image)
        
        # Read text
        results = self.ocr_reader.readtext(image_array)
        
        # Extract text and calculate average confidence
        texts = []
        confidences = []
        
        for detection in results:
            text = detection[1]
            confidence = detection[2]
            if text.strip():
                texts.append(text)
                confidences.append(confidence)
        
        extracted_text = ' '.join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        return extracted_text, avg_confidence
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        if not text:
            return ""
        
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common OCR artifacts
        text = re.sub(r'[|\[\]{}]', '', text)
        
        # Remove non-printable characters
        text = ''.join(c for c in text if c.isprintable() or c.isspace())
        
        return text.strip()
    
    def get_status(self) -> Dict[str, Any]:
        """Get image processor status"""
        self._ensure_initialized()
        
        return {
            'available': self.is_available(),
            'ocr_engine': self.ocr_engine if self.ocr_engine != 'none' else None,
            'supported_formats': ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp'] if self.is_available() else [],
            'installation_instructions': self.get_installation_instructions() if not self.is_available() else None
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if self.ocr_reader is not None:
            # EasyOCR cleanup
            del self.ocr_reader
            self.ocr_reader = None
        
        self.ocr_engine = None
        self._initialized = False
        gc.collect()
        logger.info("Image processor resources cleaned up")


# Singleton instance with lazy initialization
_image_processor: Optional[ImageProcessor] = None
_processor_lock = threading.Lock()


def get_image_processor() -> ImageProcessor:
    """Get or create image processor instance (lazy singleton)"""
    global _image_processor
    
    if _image_processor is None:
        with _processor_lock:
            if _image_processor is None:
                _image_processor = ImageProcessor(lazy_init=True)
    
    return _image_processor


def cleanup_image_processor():
    """Cleanup image processor resources"""
    global _image_processor
    
    if _image_processor is not None:
        _image_processor.cleanup()
        _image_processor = None


# Register cleanup on exit
atexit.register(cleanup_image_processor)


# Convenience functions
def process_image(image_input, input_type: str = 'file') -> ImageProcessingResult:
    """
    Process image from various input types.
    
    Args:
        image_input: Image data (file path, bytes, URL, or base64)
        input_type: Type of input ('file', 'bytes', 'url', 'base64')
    
    Returns:
        ImageProcessingResult with extracted text
    """
    processor = get_image_processor()
    
    if input_type == 'file':
        return processor.process_image_file(image_input)
    elif input_type == 'bytes':
        return processor.process_image_bytes(image_input)
    elif input_type == 'url':
        return processor.process_image_url(image_input)
    elif input_type == 'base64':
        return processor.process_base64_image(image_input)
    else:
        return ImageProcessingResult(
            success=False,
            error_message=f"Unknown input type: {input_type}"
        )
