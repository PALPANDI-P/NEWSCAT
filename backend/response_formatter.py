"""
NEWSCAT Response Formatter
Professional, standardized response structure for all API endpoints
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from enum import Enum


class ResponseStatus(Enum):
    """Standard response statuses"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PARTIAL = "partial_success"


class ConfidenceLevel(Enum):
    """Confidence level categories"""
    VERY_HIGH = "very_high"      # 90-100%
    HIGH = "high"                # 70-89%
    MODERATE = "moderate"        # 50-69%
    LOW = "low"                  # 30-49%
    VERY_LOW = "very_low"        # <30%


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
    meta: Dict[str, Any] = None
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
    response = {
        "status": ResponseStatus.SUCCESS.value,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "data": {
            # Primary classification result
            "category": data.get("category", "unknown"),
            "category_display": data.get("category_display", data.get("category", "Unknown")),
            "confidence": min(100, max(0, data.get("confidence", 0))),  # Ensure 0-100
            "confidence_level": get_confidence_level(data.get("confidence", 0)),
            
            # Processing info
            "processing_time_ms": round(processing_time_ms, 2),
            "model_name": data.get("model_name", "NewsCAT"),
            "model_version": data.get("model_version", "7.0"),
            "input_type": data.get("input_type", "text"),
        }
    }
    
    # Add alternative predictions if available (up to 50+ topics)
    if "top_predictions" in data:
        response["data"]["analysis"] = {
            "top_predictions": [
                {
                    "rank": idx + 1,
                    "category": pred.get("category", ""),
                    "category_display": pred.get("category_display", pred.get("category", "")),
                    "confidence": min(100, max(0, pred.get("confidence", 0))),
                    "confidence_level": get_confidence_level(pred.get("confidence", 0))
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
            "word_count": data.get("word_count", 0)
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
        response["data"]["extracted_content"] = {
            "text": data.get("extracted_text", "")[:1000],  # First 1000 chars
            "preview_length": min(1000, len(data.get("extracted_text", ""))),
            "ocr_confidence": round(data.get("ocr_confidence", 0), 3)
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
    details: Dict[str, Any] = None
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
        "error": {
            "code": error_code,
            "message": message,
            "http_status": status_code
        }
    }
    
    if details:
        response["error"]["details"] = details
    
    return response, status_code


def create_partial_response(
    data: Dict[str, Any],
    message: str,
    warnings: List[str] = None,
    processing_time_ms: float = 0
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
        **kwargs
    }
    
    response = create_success_response(
        data,
        processing_time_ms=processing_time_ms
    )
    
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
    uptime_seconds: float = 0
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
                "status": "operational" if classifications_available else "offline"
            },
            "image_processing": {
                "available": image_processing_available,
                "status": "operational" if image_processing_available else "not_installed"
            },
            "audio_processing": {
                "available": audio_processing_available,
                "status": "operational" if audio_processing_available else "not_installed"
            },
            "video_processing": {
                "available": video_processing_available,
                "status": "operational" if video_processing_available else "not_installed"
            }
        }
    }


def format_categories_response(categories: Dict[str, str]) -> Dict[str, Any]:
    """Format categories list response"""
    formatted_categories = [
        {
            "id": cat_id,
            "name": cat_display,
            "slug": cat_id.replace("_", "-")
        }
        for cat_id, cat_display in sorted(categories.items())
    ]
    
    return {
        "status": "success",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "total_categories": len(formatted_categories),
        "categories": formatted_categories
    }


def format_model_info(
    name: str,
    version: str,
    categories: List[str],
    accuracy: float = 0,
    trained: bool = False,
    avg_inference_time_ms: float = 0,
    model_type: str = "rule-based"
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
            "accuracy_score": round(accuracy, 3) if accuracy > 0 else None,
            "average_inference_time_ms": round(avg_inference_time_ms, 2),
            "supported_categories": len(categories),
            "categories_sample": sorted(categories)[:5]
        }
    }
