"""
NEWSCAT Response Formatter
Professional, standardized response structure for all API endpoints
"""

from typing import Dict, Any, List
from datetime import datetime
from enum import Enum
import re
import uuid

# CategoryKnowledgeGraph imported locally in create_success_response to avoid circular dependencies


class ResponseStatus(Enum):
    """Standard response statuses"""

    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PARTIAL = "partial_success"


class ConfidenceLevel(Enum):
    """Confidence level categories"""

    VERY_HIGH = "very_high"  # 90-100%
    HIGH = "high"  # 70-89%
    MODERATE = "moderate"  # 50-69%
    LOW = "low"  # 30-49%
    VERY_LOW = "very_low"  # <30%


def get_confidence_level(confidence: float) -> str:
    """Convert confidence score to level string"""
    if confidence >= 90:
        return ConfidenceLevel.VERY_HIGH.value
    elif confidence >= 70:
        return ConfidenceLevel.HIGH.value
    elif confidence >= 50:
        return ConfidenceLevel.MODERATE.value
    elif confidence >= 30:
        return ConfidenceLevel.LOW.value
    else:
        return ConfidenceLevel.VERY_LOW.value


def create_success_response(
    data: Dict[str, Any],
    message: str = None,
    processing_time_ms: float = 0,
    meta: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Create a standardized success response

    Args:
        data: Classification data
        message: Optional message
        processing_time_ms: Processing time in milliseconds
        meta: Additional metadata

    Returns:
        Formatted response dictionary
    """
    # Derive main_topic and subtopic from Knowledge Graph if not explicitly provided
    category = data.get("category", "unknown")
    main_topic = data.get("main_topic")
    subtopic = data.get("subtopic")

    try:
        from backend.models.lightning_classifier import CategoryKnowledgeGraph
    except ImportError:
        CategoryKnowledgeGraph = None

    if CategoryKnowledgeGraph and category in CategoryKnowledgeGraph.CATEGORIES:
        parent = CategoryKnowledgeGraph.CATEGORIES[category].get("parent")
        if parent:
            main_topic = main_topic or parent
            subtopic = subtopic or category
        else:
            main_topic = main_topic or category
            subtopic = subtopic or category

    # Fallback to category itself
    main_topic = main_topic or category
    subtopic = subtopic or category

    # Build main_topics list - FILTER OUT None values to prevent display issues
    main_topics_list = data.get("main_topics")
    if main_topics_list:
        # Filter None and empty values
        main_topics_list = [t for t in main_topics_list if t]
    else:
        main_topics_list = []
    
    # Add top predictions, filtering out None categories
    top_predictions = data.get("top_predictions", [])[:5]
    pred_categories = [pred.get("category") for pred in top_predictions if pred.get("category")]
    
    # Combine and remove duplicates while preserving order
    combined_topics = main_topics_list + pred_categories
    seen = set()
    final_topics = []
    for t in combined_topics:
        if t not in seen:
            seen.add(t)
            final_topics.append(t)
    
    # Ensure at least the main_topic is included
    if not final_topics:
        final_topics = [main_topic]

    # Map model tags to expert model names
    raw_models = data.get("models_run", [data.get("input_type", "text")])
    expert_models = []
    model_map = {
        "text": "Text Heuristic Engine v9.0",
        "image": "Vision OCR Engine v8.0",
        "audio": "Whisper STT Audio Engine",
        "video": "MoviePy/OpenCV Video Engine",
    }
    for m in raw_models:
        expert_models.append(model_map.get(m, f"{m.title()} Expert Engine"))

    # Extract strings for absolute hierarchy checking
    core_topic_str = data.get("category_display", category.replace("_", " ").title())
    content_main_topic_str = main_topic.replace("_", " ").title()

    # 1. Absolute Topic Hierarchy & Professional Fallbacks
    confidence_val = data.get("confidence", 0)
    
    # If the core topic is too generic or low confidence, normalize it
    if core_topic_str.lower() == "unknown" or confidence_val < 15:
        core_topic_str = "General News"
        content_main_topic_str = "Global Information"
    
    # Ensure distinctness: If Core == Main, find a better Main
    if core_topic_str.lower().strip() == content_main_topic_str.lower().strip():
        # Consult our expert fallback umbrellas
        fallback_umbrellas = {
            "technology": "Technology & AI",
            "business": "Business & Economy",
            "health": "Health & Medicine",
            "science": "Science & Innovation",
            "politics": "Politics & Policy",
            "entertainment": "Entertainment & Media",
            "sports": "Sports & Athletics",
            "lifestyle": "Lifestyle & Society",
            "breaking_news": "Global Alerts",
            "artificial_intelligence": "Advanced Technology",
            "cricket": "Professional Sports",
            "finance": "Global Markets",
            "cybersecurity": "Digital Security",
            "geopolitics": "International Relations",
            "war_conflict": "Global Security",
            "climate_change": "Environmental Science",
            "education": "Academic & Learning",
            "crime": "Law & Order",
        }
        
        # Try to find a matched umbrella for the core category slug
        slug = category.lower().strip()
        matched = fallback_umbrellas.get(slug)
        
        if matched:
            content_main_topic_str = matched
        else:
            # If no mapping, check if it contains a keyword we know
            for key, val in fallback_umbrellas.items():
                if key in slug or slug in key:
                    content_main_topic_str = val
                    break
            else:
                # Last resort: Add a prefix
                content_main_topic_str = f"General {content_main_topic_str}"

    # Final Check: Never allow identical names
    if core_topic_str.lower().strip() == content_main_topic_str.lower().strip():
        content_main_topic_str = f"{content_main_topic_str} Category"

    # 2. Intelligent Media Summaries
    input_type = data.get("input_type", "text")
    raw_summary = str(data.get("main_topic_summary", data.get("summary", "Classification complete.")))
    
    # Strip hidden AI injection tags from the literal text output so the UI reads cleanly
    base_summary = re.sub(r'\[SCENE_OBJECTS:[^\]]*\]', '', raw_summary)
    base_summary = re.sub(r'\[ACOUSTIC_SCENE:[^\]]*\]', '', base_summary).strip()
    
    # Generate Contextual Summary Models for Media Types
    final_summary = base_summary
    if input_type == "video":
        visuals = "assorted elements"
        acoustics = "native sound"
        scene_match = re.search(r'\[SCENE_OBJECTS:\s*([^\]]+)\]', raw_summary)
        if scene_match: visuals = scene_match.group(1)
        audio_match = re.search(r'\[ACOUSTIC_SCENE:\s*([^\]]+)\]', raw_summary)
        if audio_match: acoustics = audio_match.group(1)
        final_summary = f"Video content analyzing {core_topic_str}. Visual analysis identified: {visuals}. Audio analysis detected: {acoustics}."
    elif input_type == "image":
        visuals = "assorted elements"
        scene_match = re.search(r'\[SCENE_OBJECTS:\s*([^\]]+)\]', raw_summary)
        if scene_match: visuals = scene_match.group(1)
        final_summary = f"Visual media classified as {core_topic_str}, depicting {visuals}."
    elif input_type == "audio":
        acoustics = "native sound"
        audio_match = re.search(r'\[ACOUSTIC_SCENE:\s*([^\]]+)\]', raw_summary)
        if audio_match: acoustics = audio_match.group(1)
        final_summary = f"Audio recording related to {core_topic_str}. Ambient environment indicates {acoustics}."

    response = {
        "status": ResponseStatus.SUCCESS.value,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "data": {
            # Requested Custom JSON Format (Core = Specific, Main = Broad Parent)
            "core_topic": core_topic_str,
            "content_main_topic": content_main_topic_str,
            "classification_models_used": expert_models,
            
            # Required UI Frontend Rendering Keys
            "category": category,
            "category_display": core_topic_str,
            "main_topic": main_topic,
            "subtopic": subtopic,
            "category_slug": category,
            "main_topics": final_topics,
            
            # Additional context
            "confidence": min(100, max(0, data.get("confidence", 0))),
            "confidence_level": get_confidence_level(data.get("confidence", 0)),
            # Processing info
            "processing_time_ms": round(processing_time_ms, 2),
            "model_version": data.get("model_version", "9.0"),
            "input_type": input_type,
            "main_topic_summary": final_summary,
        },
    }

    # Add alternative predictions if available (up to 50+ topics)
    if "top_predictions" in data:
        response["data"]["analysis"] = {
            "top_predictions": [
                {
                    "rank": idx + 1,
                    "category": pred.get("category", ""),
                    "category_display": pred.get(
                        "category_display", pred.get("category", "")
                    ),
                    "confidence": min(100, max(0, pred.get("confidence", 0))),
                    "confidence_level": get_confidence_level(pred.get("confidence", 0)),
                }
                for idx, pred in enumerate(data.get("top_predictions", []))
            ]
        }

    # Add text content analysis
    if "keywords" in data:
        response["data"]["analysis"] = response["data"].get("analysis", {})
        response["data"]["analysis"]["keywords"] = data.get("keywords", [])[:20]

    # Add content summary
    if "summary" in data:
        response["data"]["summary"] = data.get("summary", "")

    # Add core content metrics
    if any(k in data for k in ["sentiment", "entities_count", "main_topic"]):
        response["data"]["core_content"] = {
            "main_topic": data.get("main_topic", data.get("category", "Unknown")),
            "sentiment": data.get("sentiment", "Neutral"),
            "entities_count": data.get("entities_count", 0),
            "content_length": data.get("content_length", 0),
            "word_count": data.get("word_count", 0),
        }

    # Add content metadata for text
    if "content_length" in data:
        response["data"]["metrics"] = {
            "character_count": data.get("content_length", 0),
            "word_count": data.get("word_count", 0),
            "sentence_count": data.get("sentence_count", 0),
        }

    # Add multimedia data
    if "extracted_text" in data:
        extracted_text = data.get("extracted_text") or ""
        response["data"]["extracted_content"] = {
            "text": extracted_text[:1000],  # First 1000 chars
            "preview_length": min(1000, len(extracted_text)),
            "ocr_confidence": round(data.get("ocr_confidence", 0), 3),
        }

    if "visual_info" in data:
        response["data"]["visual_analysis"] = data.get("visual_info", {})

    # Add message if provided
    if message:
        response["message"] = message

    # Add metadata if provided
    if meta:
        response["meta"] = meta

    return response


def create_error_response(
    message: str,
    error_code: str = "UNKNOWN_ERROR",
    status_code: int = 500,
    details: Dict[str, Any] = None,
) -> tuple:
    """
    Create a standardized error response

    Args:
        message: Error message
        error_code: Error identifier
        status_code: HTTP status code
        details: Additional error details

    Returns:
        Tuple of (response_dict, status_code)
    """
    response = {
        "status": ResponseStatus.ERROR.value,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "error": {"code": error_code, "message": message, "http_status": status_code},
    }

    if details:
        response["error"]["details"] = details

    return response, status_code


def create_partial_response(
    data: Dict[str, Any],
    message: str,
    warnings: List[str] = None,
    processing_time_ms: float = 0,
) -> Dict[str, Any]:
    """
    Create a partial success response (some data available, but with warnings)

    Args:
        data: Partial classification data
        message: Status message
        warnings: List of warning messages
        processing_time_ms: Processing time in milliseconds

    Returns:
        Formatted response dictionary
    """
    response = create_success_response(data, message, processing_time_ms)
    response["status"] = ResponseStatus.PARTIAL.value

    if warnings:
        response["warnings"] = warnings

    return response


def format_classification_result(
    category: str,
    confidence: float,
    top_predictions: List[Dict] = None,
    keywords: List[str] = None,
    model_name: str = "NewsCAT",
    model_version: str = "7.0",
    input_type: str = "text",
    processing_time_ms: float = 0,
    content_metrics: Dict[str, int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Format a complete classification result with all metrics
    """
    data = {
        "category": category,
        "confidence": min(100, max(0, confidence)),
        "top_predictions": top_predictions or [],
        "keywords": keywords or [],
        "model_name": model_name,
        "model_version": model_version,
        "input_type": input_type,
        **kwargs,
    }

    response = create_success_response(data, processing_time_ms=processing_time_ms)

    # Add content metrics if provided
    if content_metrics:
        if "metrics" not in response["data"]:
            response["data"]["metrics"] = {}
        response["data"]["metrics"].update(content_metrics)

    return response


def format_health_check(
    classifications_available: bool = True,
    image_processing_available: bool = False,
    audio_processing_available: bool = False,
    video_processing_available: bool = False,
    version: str = "7.0",
    uptime_seconds: float = 0,
) -> Dict[str, Any]:
    """Format health check response"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": version,
        "uptime_seconds": round(uptime_seconds, 2),
        "capabilities": {
            "text_classification": {
                "available": classifications_available,
                "status": "operational" if classifications_available else "offline",
            },
            "image_processing": {
                "available": image_processing_available,
                "status": "operational"
                if image_processing_available
                else "not_installed",
            },
            "audio_processing": {
                "available": audio_processing_available,
                "status": "operational"
                if audio_processing_available
                else "not_installed",
            },
            "video_processing": {
                "available": video_processing_available,
                "status": "operational"
                if video_processing_available
                else "not_installed",
            },
        },
    }


def format_categories_response(categories: Dict[str, str]) -> Dict[str, Any]:
    """Format categories list response"""
    formatted_categories = [
        {"id": cat_id, "name": cat_display, "slug": cat_id.replace("_", "-")}
        for cat_id, cat_display in sorted(categories.items())
    ]

    return {
        "status": "success",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "total_categories": len(formatted_categories),
        "categories": formatted_categories,
    }


def format_model_info(
    name: str,
    version: str,
    categories: List[str],
    accuracy: float = 0,
    trained: bool = False,
    avg_inference_time_ms: float = 0,
    model_type: str = "rule-based",
) -> Dict[str, Any]:
    """Format model information response"""
    return {
        "status": "success",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": {
            "name": name,
            "version": version,
            "type": model_type,
            "status": "ready" if trained else "untrained",
            "trained": trained,
            "accuracy_score": round(accuracy, 3) if accuracy and accuracy > 0 else None,
            "average_inference_time_ms": round(avg_inference_time_ms, 2),
            "supported_categories": len(categories),
            "categories_sample": sorted(categories)[:5],
        },
    }
