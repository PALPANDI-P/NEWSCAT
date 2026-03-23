"""
NEWSCAT Utilities & Helpers
Common functions, validators, and utilities for backend services
"""

import re
import logging
from typing import Dict, Any, Optional, Tuple
from functools import lru_cache
import time

logger = logging.getLogger(__name__)


# =============================================================================
# TEXT VALIDATION & SANITIZATION
# =============================================================================


class TextValidator:
    """Validate and sanitize text inputs"""

    MIN_LENGTH = 5
    MAX_LENGTH = 100000

    FORBIDDEN_PATTERNS = [
        r"<script[^>]*>.*?</script>",  # JavaScript blocks
        r"<iframe[^>]*>.*?</iframe>",  # iFrames
        r"javascript:",  # javascript protocol
        # Optimized to avoid false positives on common news phrases (e.g. "on Tuesday =")
        r"\bon(?:click|load|submit|mouse|key)\w*\s*=", 
    ]

    @staticmethod
    def is_valid(text: str) -> Tuple[bool, str]:
        """
        Validate text input
        Returns: (is_valid, message)
        """
        if not text:
            return False, "Text is required"

        if not isinstance(text, str):
            return False, "Text must be a string"

        text = text.strip()

        if len(text) < TextValidator.MIN_LENGTH:
            return (
                False,
                f"Text too short (minimum {TextValidator.MIN_LENGTH} characters)",
            )

        if len(text) > TextValidator.MAX_LENGTH:
            return (
                False,
                f"Text too long (maximum {TextValidator.MAX_LENGTH} characters)",
            )

        # Check for forbidden patterns
        for pattern in TextValidator.FORBIDDEN_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                return False, "Text contains forbidden content"

        return True, text

    @staticmethod
    def sanitize(text: str) -> str:
        """Remove potentially dangerous content"""
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    @staticmethod
    def get_metrics(text: str) -> Dict[str, int]:
        """Get text metrics"""
        return {
            "characters": len(text),
            "words": len(text.split()),
            "sentences": len([s for s in text.split(".") if s.strip()]),
            "paragraphs": len([p for p in text.split("\n\n") if p.strip()]),
        }


# =============================================================================
# PERFORMANCE & CACHING
# =============================================================================


class PerformanceMonitor:
    """Monitor and track performance metrics"""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration = (self.end_time - self.start_time) * 1000  # Convert to ms
        logger.debug(f"{self.name} took {self.duration:.2f}ms")

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        if self.end_time is None:
            self.end_time = time.perf_counter()
        return (self.end_time - self.start_time) * 1000


class SmartCache:
    """
    Smart cache with TTL and size limits
    Implements LFU (Least Frequently Used) eviction
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_count = {}
        self.access_time = {}

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key not in self.cache:
            return None

        # Check if expired
        if time.time() - self.access_time[key] > self.ttl_seconds:
            del self.cache[key]
            del self.access_count[key]
            del self.access_time[key]
            return None

        # Update access count
        self.access_count[key] = self.access_count.get(key, 0) + 1
        return self.cache[key]

    def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        # Evict if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Remove least frequently used item
            lfu_key = min(self.access_count, key=self.access_count.get)
            del self.cache[lfu_key]
            del self.access_count[lfu_key]
            del self.access_time[lfu_key]

        self.cache[key] = value
        self.access_count[key] = 1
        self.access_time[key] = time.time()

    def clear(self) -> None:
        """Clear entire cache"""
        self.cache.clear()
        self.access_count.clear()
        self.access_time.clear()

    def size(self) -> int:
        """Get current cache size"""
        return len(self.cache)


# =============================================================================
# DATA FORMATTING & CONVERSION
# =============================================================================


class DataFormatter:
    """Standard data formatting utilities"""

    @staticmethod
    def ensure_confidence_range(confidence: float) -> float:
        """Ensure confidence is between 0-100"""
        return min(100, max(0, confidence))

    @staticmethod
    def format_confidence(confidence: float, precision: int = 1) -> str:
        """Format confidence as percentage string"""
        conf = DataFormatter.ensure_confidence_range(confidence)
        return f"{conf:.{precision}f}%"

    @staticmethod
    def normalize_category(category: str) -> str:
        """Normalize category name"""
        return category.lower().strip().replace(" ", "_")

    @staticmethod
    def humanize_category(category: str) -> str:
        """Convert category slug to human-readable name"""
        return category.replace("_", " ").title()

    @staticmethod
    def truncate(text: str, max_length: int = 500, suffix: str = "...") -> str:
        """Truncate text to max length"""
        if len(text) <= max_length:
            return text
        return text[: max_length - len(suffix)] + suffix

    @staticmethod
    def bytes_to_mb(bytes_size: int) -> float:
        """Convert bytes to megabytes"""
        return round(bytes_size / (1024 * 1024), 2)

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        else:
            minutes = seconds / 60
            return f"{minutes:.1f}m"


# =============================================================================
# LOGGING & DEBUGGING
# =============================================================================


class LogContext:
    """Context manager for structured logging"""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.context = kwargs
        self.logger = logging.getLogger(name)

    def __enter__(self):
        self.logger.info(f"Starting: {self.name}", extra=self.context)
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if exc_type is None:
            self.logger.info(
                f"Completed: {self.name}",
                extra={**self.context, "duration_ms": duration * 1000},
            )
        else:
            self.logger.error(
                f"Failed: {self.name}",
                extra={
                    **self.context,
                    "error": str(exc_val),
                    "duration_ms": duration * 1000,
                },
            )


# =============================================================================
# FILE & PATH UTILITIES
# =============================================================================


class FileUtils:
    """File and path utilities"""

    ALLOWED_EXTENSIONS = {
        "text": {"txt", "md", "json", "csv"},
        "image": {"jpg", "jpeg", "png", "gif", "webp", "bmp"},
        "audio": {"mp3", "wav", "aac", "flac", "ogg", "m4a"},
        "video": {"mp4", "avi", "mkv", "mov", "webm", "flv"},
    }

    @staticmethod
    def is_allowed_extension(filename: str, file_type: str) -> bool:
        """Check if file extension is allowed"""
        if "." not in filename:
            return False

        ext = filename.rsplit(".", 1)[1].lower()
        allowed = FileUtils.ALLOWED_EXTENSIONS.get(file_type, set())
        return ext in allowed

    @staticmethod
    @lru_cache(maxsize=128)
    def get_extension(filename: str) -> str:
        """Get file extension"""
        return filename.rsplit(".", 1)[1].lower() if "." in filename else ""

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename"""
        # Remove dangerous characters
        filename = re.sub(r"[^\w\s.-]", "", filename)
        # Replace spaces with underscores
        filename = filename.replace(" ", "_")
        return filename


# =============================================================================
# ERROR HANDLING
# =============================================================================


class ErrorHandler:
    """Centralized error handling"""

    ERROR_MESSAGES = {
        "INVALID_INPUT": "Input validation failed",
        "PROCESSING_ERROR": "Processing failed",
        "FILE_ERROR": "File operation failed",
        "MEMORY_ERROR": "Out of memory",
        "TIMEOUT_ERROR": "Operation timed out",
        "MODEL_ERROR": "Model processing failed",
        "DEPENDENCY_ERROR": "Required dependency not available",
    }

    @staticmethod
    def get_message(error_code: str, default: str = None) -> str:
        """Get error message by code"""
        return ErrorHandler.ERROR_MESSAGES.get(
            error_code, default or "An error occurred"
        )

    @staticmethod
    def log_exception(exception: Exception, context: str = None) -> Dict[str, Any]:
        """Log exception with context"""
        error_dict = {
            "type": type(exception).__name__,
            "message": str(exception),
        }

        if context:
            error_dict["context"] = context

        logger.exception(f"Exception occurred: {error_dict}")
        return error_dict


# =============================================================================
# STATISTICS & METRICS
# =============================================================================


class MetricsCollector:
    """Collect and track metrics"""

    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()

    def record(self, key: str, value: float) -> None:
        """Record a metric"""
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append(value)

    def get_stats(self, key: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        values = self.metrics.get(key, [])
        if not values:
            return {}

        return {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "count": len(values),
        }

    def get_uptime(self) -> float:
        """Get uptime in seconds"""
        return time.time() - self.start_time


# Global instances
_text_validator = TextValidator()
_smart_cache = SmartCache()
_metrics_collector = MetricsCollector()


# =============================================================================
# CONTENT SUMMARIZATION - Fast extractive summarization
# =============================================================================


class ContentSummarizer:
    """
    Fast extractive content summarizer - creates short summary from input text
    Creates actual sentence-based summaries of the content, not keyword summaries
    """

    # Words to skip in summary
    SKIP_WORDS = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "been",
        "be",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "they",
        "them",
        "their",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "him",
        "her",
        "his",
        "hers",
        "what",
        "which",
        "who",
        "whom",
        "where",
        "when",
        "why",
        "how",
        "about",
        "after",
        "before",
        "during",
        "between",
        "into",
        "through",
    }

    @staticmethod
    def summarize(text: str, max_sentences: int = 3, max_words: int = 50) -> str:
        """
        Create a SHORT SENTENCE-BASED summary of the input content
        NOT a keyword summary - actual content summary

        Args:
            text: Input text to summarize
            max_sentences: Maximum number of sentences in summary (default: 3)
            max_words: Maximum number of words in summary (default: 50)

        Returns:
            Short sentence summary like: "This article discusses X happening in Y location..."
        """
        if not text or len(text.strip()) < 30:
            return text.strip() if text else ""

        try:
            # Split into sentences
            sentences = ContentSummarizer._split_into_sentences(text)

            if not sentences:
                return (
                    text[: max_words * 5] + "..." if len(text) > max_words * 5 else text
                )

            # Score each sentence - prefer complete, informative sentences
            scored = []
            for i, sentence in enumerate(sentences):
                score = ContentSummarizer._score_for_summary(sentence)
                # Position bonus for first two sentences
                if i == 0: score *= 1.5
                if i == 1: score *= 1.2
                if score > 0:
                    scored.append((score, i, sentence))

            # Sort by score and take top sentences
            scored.sort(key=lambda x: x[0], reverse=True)

            # Get top sentences but keep them in original order for coherence
            top_sent_data = scored[:max_sentences]
            top_sent_data.sort(key=lambda x: x[1]) # sort by original index
            
            selected = [s[2] for s in top_sent_data]

            # Build summary
            summary = " ".join(selected)

            # Trim to max words
            words = summary.split()
            if len(words) > max_words:
                summary = " ".join(words[:max_words])
                if not summary.endswith((".", "!", "?")):
                    summary += "..."

            return summary.strip()

        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            return text[:150].strip() + "..."

    @staticmethod
    def _split_into_sentences(text: str) -> list:
        """Split text into meaningful sentences"""
        # Split on sentence endings with lookbehind for abbreviations
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.split()) >= 3]

    @staticmethod
    def _score_for_summary(sentence: str) -> float:
        """Score a sentence for being a good summary - prefers complete statements"""
        words = sentence.split()
        word_count = len(words)

        if word_count < 8:
            return 0

        score = 1.0

        # Prefer sentences with proper nouns (likely key info)
        has_proper_noun = any(w[0].isupper() for w in words if len(w) > 1)
        if has_proper_noun:
            score += 3.0

        # Prefer sentences with action/news verbs
        action_words = [
            "said", "announced", "reported", "stated", "revealed", "discovered",
            "launched", "introduced", "unveiled", "confirmed", "ordered",
            "decided", "increased", "decreased", "grew", "declined", "approved",
            "rejected", "winning", "losing", "match", "game", "update", "new"
        ]
        lower = sentence.lower()
        for aw in action_words:
            if aw in lower:
                score += 2.0

        # Mid-length sentences are best for summaries
        if 12 <= word_count <= 35:
            score += 4.0
        elif 8 <= word_count < 12:
            score += 2.0

        # Penalty for promotional language
        if any(w in lower for w in ["click", "subscribe", "follow", "sign up", "limited time"]):
            score -= 10.0

        return max(0, score)


def get_text_validator() -> TextValidator:
    """Get global text validator instance"""
    return _text_validator


def get_smart_cache() -> SmartCache:
    """Get global smart cache instance"""
    return _smart_cache


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    return _metrics_collector


def extract_keywords(text: str, max_keywords: int = 8) -> list:
    """Extract top keywords from text without high-level NLP libraries"""
    if not text:
        return []

    # Common stop words
    stop_words = ContentSummarizer.SKIP_WORDS | {
        "doing", "done", "used", "using", "take", "taking", "make", "making",
        "also", "could", "should", "would", "many", "most", "some", "around",
        "about", "really", "very", "much", "many", "even", "just", "then",
        "than", "now", "only", "well", "more", "most", "also", "into"
    }

    # Clean and split: only words with 3+ characters (to catch 'AI', '5G', etc. we keep 2+)
    words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    filtered = [w for w in words if w not in stop_words]

    # Count frequencies
    counts = {}
    for w in filtered:
        counts[w] = counts.get(w, 0) + 1

    # Get top keywords
    sorted_keys = sorted(counts, key=counts.get, reverse=True)
    return sorted_keys[:max_keywords]
