"""
NEWSCAT Image Processor — OCR + visual analysis for news image classification.
Supports Pillow (basic), EasyOCR, and Pytesseract backends.
Gracefully degrades when optional dependencies are unavailable.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger(__name__)


@dataclass
class ImageResult:
    """Result of image processing."""
    success: bool = False
    extracted_text: str = ""
    confidence: float = 0.0
    error_message: str = ""
    metadata: dict = field(default_factory=dict)


class ImageProcessor:
    """
    Multi-backend image processor.
    Priority: EasyOCR > Pytesseract > Pillow-only (no OCR).
    """

    def __init__(self):
        self._pil_available = False
        self._ocr_engine = "none"
        self._easyocr_reader = None
        self._detect_backends()

    def _detect_backends(self):
        """Detect available image processing backends."""
        # Check Pillow
        try:
            from PIL import Image  # noqa: F401
            self._pil_available = True
        except ImportError:
            logger.warning("Pillow not installed — image loading unavailable")

        # Check EasyOCR
        try:
            import easyocr  # noqa: F401
            self._ocr_engine = "easyocr"
            logger.info("EasyOCR detected — using as primary OCR engine")
            return
        except ImportError:
            pass

        # Check Pytesseract
        try:
            import pytesseract  # noqa: F401
            self._ocr_engine = "pytesseract"
            logger.info("Pytesseract detected — using as OCR engine")
            return
        except ImportError:
            pass

        if self._pil_available:
            logger.info("No OCR engine found — basic image processing only")
        else:
            logger.warning("No image processing libraries available")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """True if at least Pillow is installed."""
        return self._pil_available

    def is_pil_available(self) -> bool:
        return self._pil_available

    @property
    def ocr_engine(self) -> str:
        return self._ocr_engine

    def get_installation_instructions(self) -> str:
        parts = []
        if not self._pil_available:
            parts.append("pip install Pillow")
        if self._ocr_engine == "none":
            parts.append("pip install easyocr  # OR  pip install pytesseract")
        return " && ".join(parts) if parts else "All dependencies installed"

    def process_image_file(self, file_path: str) -> ImageResult:
        """
        Extract text from an image file using the best available OCR engine.
        """
        if not os.path.exists(file_path):
            return ImageResult(success=False, error_message=f"File not found: {file_path}")

        if not self._pil_available:
            return ImageResult(
                success=False,
                error_message="Pillow is not installed. Run: pip install Pillow",
            )

        try:
            from PIL import Image
            img = Image.open(file_path)
            width, height = img.size
            metadata = {
                "width": width,
                "height": height,
                "format": img.format or "unknown",
                "mode": img.mode,
            }
        except Exception as e:
            return ImageResult(success=False, error_message=f"Cannot open image: {e}")

        # Try OCR
        extracted = ""
        confidence = 0.0

        if self._ocr_engine == "easyocr":
            extracted, confidence = self._ocr_easyocr(file_path)
        elif self._ocr_engine == "pytesseract":
            extracted, confidence = self._ocr_pytesseract(file_path)

        # If OCR produced nothing, try basic Pillow analysis
        if not extracted.strip():
            extracted = self._basic_analysis(img)
            confidence = 0.3 if extracted else 0.1
            
        # ------------------------------------------------------------------
        # EXPERT NOISE FILTER (Real-World Accuracy Boost)
        # ------------------------------------------------------------------
        if extracted.strip():
            import re
            # Remove non-alphanumeric symbol spam
            extracted = re.sub(r'[^\w\s\.]', ' ', extracted)
            # Remove 1-2 letter garbage chunks (unless they are specific keywords, but usually they are noise)
            extracted = re.sub(r'\b\w{1,2}\b', '', extracted)
            # Collapse whitespace
            extracted = re.sub(r'\s+', ' ', extracted).strip()
            logger.debug(f"OCR Noise Filter applied. Remaining text: {extracted[:100]}...")

        # ------------------------------------------------------------------
        # EXPERT VISUAL SCENE AI (MobileNetV2 Zero-Shot Classification)
        # ------------------------------------------------------------------
        scene_tags = []
        try:
            import torch
            from transformers import pipeline
            if not hasattr(self, "_scene_pipeline"):
                # Load ultra-lightweight mobilenet model (approx 14MB) for CPU
                self._scene_pipeline = pipeline(
                    "image-classification", 
                    model="google/mobilenet_v2_1.0_224",
                    device=-1 # Force CPU
                )
            
            # Predict top 3 objects/scenes in the photo
            predictions = self._scene_pipeline(img, top_k=3)
            for p in predictions:
                if p["score"] > 0.1:
                    # Clean and format the ImageNet label (e.g. "space shuttle" -> "space shuttle")
                    labels = p["label"].split(",")
                    scene_tags.extend([l.strip().lower() for l in labels])
            
            if scene_tags:
                metadata["visual_scene_tags"] = scene_tags
                # Inject the visual objects into the OCR text so the text classifier scores the physical image content!
                scene_string = " ".join(scene_tags)
                extracted = f"{extracted} [SCENE_OBJECTS: {scene_string}]"
                
                # Boost confidence wildly since we definitely identified objects
                confidence = min(0.95, confidence + 0.5)
                logger.debug(f"A.I. Visual Scene mapped objects: {scene_string}")
                
        except ImportError:
            logger.debug("Transformers/Torch not installed; skipping Expert Visual Scene AI.")
        except Exception as e:
            logger.debug(f"Expert Visual Scene AI skipped during processing: {e}")

        return ImageResult(
            success=True,
            extracted_text=extracted.strip(),
            confidence=confidence,
            metadata=metadata,
        )

    def process_image_data(self, image_data: bytes) -> ImageResult:
        """Process image from raw bytes."""
        import tempfile
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(image_data)
                tmp_path = tmp.name
            result = self.process_image_file(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        return result

    # ------------------------------------------------------------------
    # OCR Backends
    # ------------------------------------------------------------------

    def _ocr_easyocr(self, file_path: str) -> tuple:
        """Extract text using EasyOCR."""
        try:
            import easyocr
            if self._easyocr_reader is None:
                self._easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            results = self._easyocr_reader.readtext(file_path)
            texts = [r[1] for r in results if r[2] > 0.2]
            avg_conf = sum(r[2] for r in results) / len(results) if results else 0
            return " ".join(texts), avg_conf
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return "", 0.0

    def _ocr_pytesseract(self, file_path: str) -> tuple:
        """Extract text using Pytesseract."""
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
            # Pytesseract doesn't return confidence per-word easily,
            # so we estimate based on text length
            confidence = min(0.9, 0.3 + len(text.split()) * 0.02)
            return text, confidence
        except Exception as e:
            logger.error(f"Pytesseract failed: {e}")
            return "", 0.0

    def _basic_analysis(self, img) -> str:
        """Basic image analysis without OCR — returns empty for now."""
        try:
            # Could do colour histogram / edge analysis in future
            return ""
        except Exception:
            return ""


# ------------------------------------------------------------------
# Singleton factory
# ------------------------------------------------------------------

_image_processor: Optional[ImageProcessor] = None


def get_image_processor() -> ImageProcessor:
    global _image_processor
    if _image_processor is None:
        _image_processor = ImageProcessor()
    return _image_processor
