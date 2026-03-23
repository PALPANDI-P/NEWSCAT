import time
import json
import logging
import hashlib
from typing import Any, Dict, Optional, Callable
from functools import wraps

logger = logging.getLogger(__name__)

class AsyncCache:
    """
    Simple asynchronous in-memory LRU-like cache.
    Stores results based on a key (usually a hash of the input).
    """
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl # Time to live in seconds

    def _get_hash(self, key: str) -> str:
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    async def get(self, key: str) -> Optional[Any]:
        """Retrieve an item from the cache if it hasn't expired."""
        h = self._get_hash(key)
        if h in self._cache:
            item = self._cache[h]
            if time.time() < item["expires_at"]:
                logger.debug(f"Cache hit for key hash: {h}")
                # Update last accessed for LRU logic (simplified)
                item["last_accessed"] = time.time()
                return item["value"]
            else:
                # Expired
                logger.debug(f"Cache expired for key hash: {h}")
                del self._cache[h]
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Store an item in the cache."""
        h = self._get_hash(key)
        
        # Eviction policy: if cache full, remove the least recently accessed item
        if len(self._cache) >= self.max_size:
            lru_key = min(self._cache.keys(), key=lambda k: self._cache[k]["last_accessed"])
            del self._cache[lru_key]
            logger.debug(f"Cache full, evicted LRU item: {lru_key}")

        ttl = ttl or self.default_ttl
        self._cache[h] = {
            "value": value,
            "expires_at": time.time() + ttl,
            "last_accessed": time.time()
        }
        logger.debug(f"Cache set for key hash: {h}")

    def clear(self):
        """Clear the entire cache."""
        self._cache.clear()

# Global cache instance
_global_cache = AsyncCache()

def get_cache() -> AsyncCache:
    return _global_cache
