"""
NEWSCAT Utilities & Helpers
Common functions, validators, and utilities for backend services
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from functools import lru_cache
import time

logger = logging.getLogger(__name__)


# =============================================================================
# TEXT VALIDATION & SANITIZATION
# =============================================================================

class TextValidator:
    """Validate and sanitize text inputs"""
    
    MIN_LENGTH = 5
    MAX_LENGTH = 50000
    
    FORBIDDEN_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # JavaScript
        r'<iframe[^>]*>.*?</iframe>',  # iFrames
        r'javascript:',                 # javascript protocol
        r'on\w+\s*=',                   # Event handlers
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
            return False, f"Text too short (minimum {TextValidator.MIN_LENGTH} characters)"
        
        if len(text) > TextValidator.MAX_LENGTH:
            return False, f"Text too long (maximum {TextValidator.MAX_LENGTH} characters)"
        
        # Check for forbidden patterns
        for pattern in TextValidator.FORBIDDEN_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                return False, "Text contains forbidden content"
        
        return True, text
    
    @staticmethod
    def sanitize(text: str) -> str:
        """Remove potentially dangerous content"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def get_metrics(text: str) -> Dict[str, int]:
        """Get text metrics"""
        return {
            'characters': len(text),
            'words': len(text.split()),
            'sentences': len([s for s in text.split('.') if s.strip()]),
            'paragraphs': len([p for p in text.split('\n\n') if p.strip()])
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
        return category.lower().strip().replace(' ', '_')
    
    @staticmethod
    def humanize_category(category: str) -> str:
        """Convert category slug to human-readable name"""
        return category.replace('_', ' ').title()
    
    @staticmethod
    def truncate(text: str, max_length: int = 500, suffix: str = "...") -> str:
        """Truncate text to max length"""
        if len(text) <= max_length:
            return text
        return text[:max_length - len(suffix)] + suffix
    
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
                extra={**self.context, 'duration_ms': duration * 1000}
            )
        else:
            self.logger.error(
                f"Failed: {self.name}",
                extra={
                    **self.context,
                    'error': str(exc_val),
                    'duration_ms': duration * 1000
                }
            )


# =============================================================================
# FILE & PATH UTILITIES
# =============================================================================

class FileUtils:
    """File and path utilities"""
    
    ALLOWED_EXTENSIONS = {
        'text': {'txt', 'md', 'json', 'csv'},
        'image': {'jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp'},
        'audio': {'mp3', 'wav', 'aac', 'flac', 'ogg', 'm4a'},
        'video': {'mp4', 'avi', 'mkv', 'mov', 'webm', 'flv'},
    }
    
    @staticmethod
    def is_allowed_extension(filename: str, file_type: str) -> bool:
        """Check if file extension is allowed"""
        if '.' not in filename:
            return False
        
        ext = filename.rsplit('.', 1)[1].lower()
        allowed = FileUtils.ALLOWED_EXTENSIONS.get(file_type, set())
        return ext in allowed
    
    @staticmethod
    @lru_cache(maxsize=128)
    def get_extension(filename: str) -> str:
        """Get file extension"""
        return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename"""
        # Remove dangerous characters
        filename = re.sub(r'[^\w\s.-]', '', filename)
        # Replace spaces with underscores
        filename = filename.replace(' ', '_')
        return filename


# =============================================================================
# ERROR HANDLING
# =============================================================================

class ErrorHandler:
    """Centralized error handling"""
    
    ERROR_MESSAGES = {
        'INVALID_INPUT': 'Input validation failed',
        'PROCESSING_ERROR': 'Processing failed',
        'FILE_ERROR': 'File operation failed',
        'MEMORY_ERROR': 'Out of memory',
        'TIMEOUT_ERROR': 'Operation timed out',
        'MODEL_ERROR': 'Model processing failed',
        'DEPENDENCY_ERROR': 'Required dependency not available',
    }
    
    @staticmethod
    def get_message(error_code: str, default: str = None) -> str:
        """Get error message by code"""
        return ErrorHandler.ERROR_MESSAGES.get(
            error_code,
            default or 'An error occurred'
        )
    
    @staticmethod
    def log_exception(exception: Exception, context: str = None) -> Dict[str, Any]:
        """Log exception with context"""
        error_dict = {
            'type': type(exception).__name__,
            'message': str(exception),
        }
        
        if context:
            error_dict['context'] = context
        
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
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'count': len(values),
        }
    
    def get_uptime(self) -> float:
        """Get uptime in seconds"""
        return time.time() - self.start_time


# Global instances
_text_validator = TextValidator()
_smart_cache = SmartCache()
_metrics_collector = MetricsCollector()


def get_text_validator() -> TextValidator:
    """Get global text validator instance"""
    return _text_validator


def get_smart_cache() -> SmartCache:
    """Get global smart cache instance"""
    return _smart_cache


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    return _metrics_collector
