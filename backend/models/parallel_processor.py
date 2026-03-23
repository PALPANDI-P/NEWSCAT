"""
NEWSCAT Parallel Processor — Async multi-modal news classification engine.
Routes text, image, audio, and video inputs to their respective workers,
runs them concurrently, and aggregates results via weighted voting.
"""

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Unified classification result from one or more modalities."""
    primary_category: str = "technology"
    confidence: float = 0.0  # 0.0 – 1.0
    model_results: Dict[str, Any] = field(default_factory=dict)


class ParallelProcessor:
    """
    Async engine that dispatches classification tasks to modality-specific
    workers and aggregates their results.
    """

    def __init__(self):
        self._classifier = None
        logger.info("ParallelProcessor initialized")

    def _get_classifier(self):
        """Lazy-load the text classifier."""
        if self._classifier is None:
            from backend.models.simple_classifier import SimpleNewsClassifier
            self._classifier = SimpleNewsClassifier()
        return self._classifier

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    async def process(
        self,
        text: Optional[str] = None,
        image_data: Optional[bytes] = None,
        audio_data: Optional[bytes] = None,
        video_data: Optional[bytes] = None,
        models: Optional[List[str]] = None,
        **kwargs,
    ) -> ClassificationResult:
        """
        Process one or more modalities in parallel.

        Parameters
        ----------
        text        : raw text to classify
        image_data  : raw image bytes
        audio_data  : raw audio bytes
        video_data  : raw video bytes
        models      : explicit list of models to run (e.g. ["text", "image"])
        **kwargs    : image_filename, audio_filename, video_filename for context

        Returns
        -------
        ClassificationResult with aggregated primary_category and confidence.
        """
        tasks = {}

        # Determine which workers to run
        if models:
            run_models = set(models)
        else:
            run_models = set()
            if text:
                run_models.add("text")
            if image_data:
                run_models.add("image")
            if audio_data:
                run_models.add("audio")
            if video_data:
                run_models.add("video")

        if not run_models:
            return ClassificationResult()

        # Create coroutine tasks
        if "text" in run_models and text:
            tasks["text"] = asyncio.create_task(
                self._text_worker(text)
            )
        if "image" in run_models and image_data:
            tasks["image"] = asyncio.create_task(
                self._image_worker(
                    image_data, kwargs.get("image_filename", "")
                )
            )
        if "audio" in run_models and audio_data:
            tasks["audio"] = asyncio.create_task(
                self._audio_worker(
                    audio_data, kwargs.get("audio_filename", "")
                )
            )
        if "video" in run_models and video_data:
            tasks["video"] = asyncio.create_task(
                self._video_worker(
                    video_data, kwargs.get("video_filename", "")
                )
            )

        # Await all tasks with a strict timeout to meet the < 5s requirement
        model_results: Dict[str, Any] = {}
        for model_name, task in tasks.items():
            try:
                # 4.5s timeout per worker (total parallel time remains ~4.5s)
                model_results[model_name] = await asyncio.wait_for(task, timeout=4.5)
            except asyncio.TimeoutError:
                logger.warning(f"Worker {model_name} timed out after 4.5s")
                # Fallback for timeout
                model_results[model_name] = {
                    "category": "technology", # General fallback
                    "confidence": 0.1,
                    "timeout": True
                }
            except Exception as e:
                logger.error(f"Worker {model_name} failed: {e}")
                model_results[model_name] = {
                    "category": "technology",
                    "confidence": 0.0,
                    "error": str(e),
                }

        # Aggregate results
        return self._aggregate(model_results)

    # ------------------------------------------------------------------
    # WORKERS
    # ------------------------------------------------------------------

    async def _text_worker(self, text: str) -> Dict[str, Any]:
        """Classify raw text."""
        loop = asyncio.get_event_loop()
        classifier = self._get_classifier()
        result = await loop.run_in_executor(
            None,
            lambda: classifier.classify(text, include_confidence=True, include_all_scores=True),
        )
        return {
            "category": result["category"],
            "confidence": result["confidence"] / 100.0,  # normalise 0-1
            "top_predictions": result.get("top_predictions", []),
        }

    async def _image_worker(
        self, image_data: bytes, filename: str = ""
    ) -> Dict[str, Any]:
        """Extract text from image, then classify."""
        from backend.models.image_processor import get_image_processor

        processor = get_image_processor()
        loop = asyncio.get_event_loop()

        # Process image
        result = await loop.run_in_executor(
            None, lambda: processor.process_image_data(image_data)
        )

        extracted_text = result.extracted_text if result.success else ""

        # Enrich with filename keywords
        if filename:
            fname_words = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")
            classification_input = f"{fname_words} {extracted_text}".strip()
        else:
            classification_input = extracted_text

        if not classification_input or len(classification_input.strip()) < 3:
            return {"category": "technology", "confidence": 0.1}

        # Classify
        classifier = self._get_classifier()
        cls_result = await loop.run_in_executor(
            None,
            lambda: classifier.classify(
                classification_input, include_confidence=True, include_all_scores=True
            ),
        )

        return {
            "category": cls_result["category"],
            "confidence": cls_result["confidence"] / 100.0,
            "extracted_text": extracted_text[:500],
        }

    async def _audio_worker(
        self, audio_data: bytes, filename: str = ""
    ) -> Dict[str, Any]:
        """Transcribe audio, then classify."""
        from backend.models.audio_processor import get_audio_processor

        processor = get_audio_processor()
        loop = asyncio.get_event_loop()

        result = await loop.run_in_executor(
            None, lambda: processor.process_audio_data(audio_data, filename)
        )

        transcribed = result.transcribed_text if result.success else ""

        # Filename enrichment
        if filename:
            fname_words = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")
            classification_input = f"{fname_words} {transcribed}".strip()
        else:
            classification_input = transcribed

        if not classification_input or len(classification_input.strip()) < 3:
            return {"category": "technology", "confidence": 0.1}

        classifier = self._get_classifier()
        cls_result = await loop.run_in_executor(
            None,
            lambda: classifier.classify(
                classification_input, include_confidence=True, include_all_scores=True
            ),
        )

        return {
            "category": cls_result["category"],
            "confidence": cls_result["confidence"] / 100.0,
            "transcribed_text": transcribed[:500],
        }

    async def _video_worker(
        self, video_data: bytes, filename: str = ""
    ) -> Dict[str, Any]:
        """Process video (audio + frames), then classify."""
        from backend.models.video_processor import get_video_processor

        processor = get_video_processor()
        loop = asyncio.get_event_loop()

        result = await loop.run_in_executor(
            None, lambda: processor.process_video_data(video_data, filename)
        )

        # Perform independent classification to avoid dilution
        classifier = self._get_classifier()
        loop = asyncio.get_event_loop()
        fname_context = ""
        if filename:
            fname_context = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")

        # 1. Classify Audio Track (Primary Signal)
        audio_result = {"category": "unknown", "confidence": 0.0}
        if result.transcribed_audio:
            audio_input = f"{fname_context} {result.transcribed_audio}".strip()
            audio_result = await loop.run_in_executor(
                None, lambda: classifier.classify(audio_input, include_confidence=True)
            )

        # 2. Classify Frame Extracts (Secondary Signal)
        frame_result = {"category": "unknown", "confidence": 0.0}
        if result.success and result.extracted_text:
            frame_result = await loop.run_in_executor(
                None, lambda: classifier.classify(f"{fname_context} {result.extracted_text}".strip(), include_confidence=True)
            )

        # 3. Pick the winner based on highest confidence
        if audio_result["confidence"] >= frame_result["confidence"]:
            winner = audio_result
        else:
            winner = frame_result
            
        # Standard fallback if both are extremely low
        if winner["confidence"] < 5.0 and not result.success:
            winner = {"category": "technology", "confidence": 5.0}

        return {
            "category": winner["category"],
            "confidence": winner["confidence"] / 100.0,
            "transcribed_text": result.transcribed_audio[:500] if result.transcribed_audio else "",
        }

    # ------------------------------------------------------------------
    # AGGREGATION
    # ------------------------------------------------------------------

    # Weight each modality by reliability
    _MODEL_WEIGHTS = {
        "text": 1.0,
        "audio": 0.85,
        "image": 0.7,
        "video": 0.9,
    }

    def _aggregate(self, model_results: Dict[str, Any]) -> ClassificationResult:
        """Weighted voting across modality results."""
        if not model_results:
            return ClassificationResult()

        # Single model — return directly
        if len(model_results) == 1:
            key = list(model_results.keys())[0]
            r = model_results[key]
            return ClassificationResult(
                primary_category=r.get("category", "technology"),
                confidence=r.get("confidence", 0.0),
                model_results=model_results,
            )

        # Multiple models — weighted vote
        category_scores: Dict[str, float] = {}
        for model_name, r in model_results.items():
            cat = r.get("category", "technology")
            conf = r.get("confidence", 0.0)
            weight = self._MODEL_WEIGHTS.get(model_name, 0.5)
            score = conf * weight
            category_scores[cat] = category_scores.get(cat, 0) + score

        # Pick winner
        if category_scores:
            best_cat = max(category_scores, key=category_scores.get)
            total_weight = sum(
                self._MODEL_WEIGHTS.get(m, 0.5) for m in model_results
            )
            best_conf = category_scores[best_cat] / total_weight if total_weight else 0
        else:
            best_cat = "technology"
            best_conf = 0.0

        return ClassificationResult(
            primary_category=best_cat,
            confidence=min(1.0, best_conf),
            model_results=model_results,
        )


# ------------------------------------------------------------------
# Singleton factory
# ------------------------------------------------------------------

_processor: Optional[ParallelProcessor] = None


def get_processor() -> ParallelProcessor:
    global _processor
    if _processor is None:
        _processor = ParallelProcessor()
    return _processor
