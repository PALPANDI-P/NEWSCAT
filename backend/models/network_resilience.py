"""
Network Resilience Module for NEWSCAT
=====================================
Provides robust network operations with:
- Connection pooling
- Retry mechanisms with exponential backoff
- Timeout handling
- Circuit breaker pattern

Author: NEWSCAT Team
Version: 1.0.0
"""

import urllib3
import logging
import time
import threading
from typing import Optional, Dict, Any, Callable, TypeVar
from functools import wraps
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Type variable for generic functions
T = TypeVar('T')

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    'pool_num_pools': 10,
    'pool_maxsize': 20,
    'retry_total': 3,
    'retry_backoff_factor': 0.5,
    'retry_status_forcelist': [500, 502, 503, 504],
    'timeout_connect': 10.0,
    'timeout_read': 30.0,
    'circuit_breaker_failure_threshold': 5,
    'circuit_breaker_recovery_timeout': 60.0,
}


class CircuitBreakerState:
    """Circuit breaker state management"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open
        self._lock = threading.Lock()
    
    def record_success(self):
        """Record a successful operation"""
        with self._lock:
            self.failure_count = 0
            self.state = "closed"
    
    def record_failure(self):
        """Record a failed operation"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def can_execute(self) -> bool:
        """Check if execution is allowed"""
        with self._lock:
            if self.state == "closed":
                return True
            
            if self.state == "open":
                # Check if recovery timeout has passed
                if self.last_failure_time and \
                   (time.time() - self.last_failure_time) > self.recovery_timeout:
                    self.state = "half-open"
                    return True
                return False
            
            # Half-open state - allow one attempt
            return True
    
    def get_state(self) -> str:
        """Get current circuit breaker state"""
        with self._lock:
            return self.state


class NetworkResilience:
    """
    Network resilience manager with connection pooling, retry logic,
    timeouts, and circuit breaker pattern.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, config: Optional[Dict] = None):
        """Singleton pattern for network resilience manager"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the network resilience manager"""
        if self._initialized:
            return
        
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self._initialized = True
        
        # Initialize connection pool
        self._pool = urllib3.PoolManager(
            num_pools=self.config['pool_num_pools'],
            maxsize=self.config['pool_maxsize'],
            block=False
        )
        
        # Initialize retry strategy
        self._retry_strategy = urllib3.util.retry.Retry(
            total=self.config['retry_total'],
            backoff_factor=self.config['retry_backoff_factor'],
            status_forcelist=self.config['retry_status_forcelist'],
            raise_on_status=False
        )
        
        # Initialize timeout
        self._timeout = urllib3.util.Timeout(
            connect=self.config['timeout_connect'],
            read=self.config['timeout_read']
        )
        
        # Circuit breakers for different services
        self._circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self._circuit_breaker_lock = threading.Lock()
        
        logger.info("NetworkResilience initialized with config: %s", self.config)
    
    def get_pool(self) -> urllib3.PoolManager:
        """Get the connection pool manager"""
        return self._pool
    
    def get_retry(self) -> urllib3.util.retry.Retry:
        """Get the retry strategy"""
        return self._retry_strategy
    
    def get_timeout(self) -> urllib3.util.Timeout:
        """Get the default timeout"""
        return self._timeout
    
    def get_circuit_breaker(self, service_name: str) -> CircuitBreakerState:
        """Get or create a circuit breaker for a service"""
        with self._circuit_breaker_lock:
            if service_name not in self._circuit_breakers:
                self._circuit_breakers[service_name] = CircuitBreakerState(
                    failure_threshold=self.config['circuit_breaker_failure_threshold'],
                    recovery_timeout=self.config['circuit_breaker_recovery_timeout']
                )
            return self._circuit_breakers[service_name]
    
    def make_request(
        self,
        method: str,
        url: str,
        retries: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Optional[urllib3.response.HTTPResponse]:
        """
        Make an HTTP request with resilience features.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: URL to request
            retries: Number of retries (uses default if None)
            timeout: Request timeout in seconds
            **kwargs: Additional arguments for request
            
        Returns:
            HTTP response or None on failure
        """
        retry_strategy = self._retry_strategy
        if retries is not None:
            retry_strategy = urllib3.util.retry.Retry(
                total=retries,
                backoff_factor=self.config['retry_backoff_factor'],
                status_forcelist=self.config['retry_status_forcelist']
            )
        
        request_timeout = timeout or self._timeout
        
        try:
            response = self._pool.request(
                method=method,
                url=url,
                timeout=request_timeout,
                retries=retry_strategy,
                **kwargs
            )
            return response
        except urllib3.exceptions.HTTPError as e:
            logger.error(f"HTTP error during request to {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during request to {url}: {e}")
            return None
    
    def get(self, url: str, **kwargs) -> Optional[urllib3.response.HTTPResponse]:
        """Make a GET request"""
        return self.make_request("GET", url, **kwargs)
    
    def post(self, url: str, **kwargs) -> Optional[urllib3.response.HTTPResponse]:
        """Make a POST request"""
        return self.make_request("POST", url, **kwargs)
    
    def close(self):
        """Close all connections and clean up"""
        if self._pool:
            self._pool.clear()
        logger.info("NetworkResilience closed")


def with_circuit_breaker(service_name: str):
    """
    Decorator to add circuit breaker functionality to a function.
    
    Args:
        service_name: Name of the service for circuit breaker
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            resilience = NetworkResilience()
            circuit_breaker = resilience.get_circuit_breaker(service_name)
            
            if not circuit_breaker.can_execute():
                logger.warning(f"Circuit breaker open for {service_name}, skipping execution")
                raise CircuitBreakerOpenError(f"Circuit breaker is open for {service_name}")
            
            try:
                result = func(*args, **kwargs)
                circuit_breaker.record_success()
                return result
            except Exception as e:
                circuit_breaker.record_failure()
                logger.error(f"Failure in {service_name}: {e}")
                raise
        
        return wrapper
    return decorator


def with_retry(
    max_retries: int = 3,
    backoff_factor: float = 0.5,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator to add retry logic to a function.
    
    Args:
        max_retries: Maximum number of retries
        backoff_factor: Exponential backoff factor
        exceptions: Tuple of exceptions to catch
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            retry_strategy = urllib3.util.retry.Retry(
                total=max_retries,
                backoff_factor=backoff_factor,
                raise_on_status=False
            )
            
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        sleep_time = backoff_factor * (2 ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {sleep_time:.2f}s..."
                        )
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            
            raise last_exception
        
        return wrapper
    return decorator


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_resilient_client(config: Optional[Dict] = None) -> NetworkResilience:
    """
    Factory function to create a resilient network client.
    
    Args:
        config: Optional configuration override
        
    Returns:
        NetworkResilience instance
    """
    return NetworkResilience(config)


# Global instance for easy access
_default_resilience: Optional[NetworkResilience] = None


def get_resilience() -> NetworkResilience:
    """Get the default network resilience instance"""
    global _default_resilience
    if _default_resilience is None:
        _default_resilience = NetworkResilience()
    return _default_resilience
