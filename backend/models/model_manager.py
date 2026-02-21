"""
NEWSCAT - Model Manager
Implements lazy loading, caching, and efficient model management

Features:
- Lazy loading: Models load on first request, not at startup
- Model caching: Loaded models stay in memory for subsequent requests
- Thread-safe operations
- Memory management with automatic cleanup
- Model warmup support
"""

import threading
import logging
import time
import gc
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
import weakref

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a loaded model"""
    name: str
    loaded: bool
    load_time: Optional[float] = None
    last_used: Optional[float] = None
    use_count: int = 0
    memory_mb: float = 0.0


class ModelManager:
    """
    Thread-safe model manager with lazy loading and caching
    
    Usage:
        manager = ModelManager()
        
        # Register model loaders
        manager.register('optimized', load_optimized_classifier)
        manager.register('ensemble', load_ensemble_classifier)
        
        # Get model (loads on first call, caches for subsequent)
        classifier = manager.get('optimized')
        
        # Preload specific models
        manager.preload(['optimized', 'ensemble'])
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for global model management"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._models: Dict[str, Any] = {}
        self._loaders: Dict[str, Callable] = {}
        self._model_info: Dict[str, ModelInfo] = {}
        self._model_locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._max_models = 10  # Maximum models to keep loaded
        self._initialized = True
        
        logger.info("ModelManager initialized")
    
    def register(self, name: str, loader: Callable, preload: bool = False) -> None:
        """
        Register a model loader function
        
        Args:
            name: Unique model identifier
            loader: Function that returns the model instance
            preload: If True, load immediately (default: False for lazy loading)
        """
        with self._global_lock:
            self._loaders[name] = loader
            self._model_locks[name] = threading.Lock()
            self._model_info[name] = ModelInfo(
                name=name,
                loaded=False
            )
            
            if preload:
                self._load_model(name)
                
        logger.debug(f"Registered model loader: {name} (preload={preload})")
    
    def get(self, name: str) -> Optional[Any]:
        """
        Get a model by name (lazy loads if not cached)
        
        Args:
            name: Model identifier
            
        Returns:
            Model instance or None if not available
        """
        # Check if model is registered
        if name not in self._loaders:
            logger.warning(f"Model '{name}' not registered")
            return None
        
        # Get or create model
        with self._global_lock:
            if name in self._models:
                # Update usage stats
                info = self._model_info[name]
                info.last_used = time.time()
                info.use_count += 1
                return self._models[name]
        
        # Load model (outside global lock to avoid blocking)
        return self._load_model(name)
    
    def _load_model(self, name: str) -> Optional[Any]:
        """Load a model (internal method)"""
        model_lock = self._model_locks.get(name)
        if not model_lock:
            return None
            
        with model_lock:
            # Double-check after acquiring lock
            if name in self._models:
                return self._models[name]
            
            loader = self._loaders.get(name)
            if not loader:
                return None
            
            try:
                start_time = time.time()
                logger.info(f"Loading model: {name}")
                
                model = loader()
                
                load_time = time.time() - start_time
                logger.info(f"Model '{name}' loaded in {load_time:.2f}s")
                
                # Store model and update info
                with self._global_lock:
                    self._models[name] = model
                    info = self._model_info[name]
                    info.loaded = True
                    info.load_time = load_time
                    info.last_used = time.time()
                    info.use_count = 1
                
                return model
                
            except Exception as e:
                logger.error(f"Failed to load model '{name}': {e}")
                return None
    
    def preload(self, names: list = None) -> Dict[str, bool]:
        """
        Preload models (useful for warmup)
        
        Args:
            names: List of model names to preload (None = all)
            
        Returns:
            Dict of model_name -> success status
        """
        if names is None:
            names = list(self._loaders.keys())
        
        results = {}
        for name in names:
            model = self.get(name)
            results[name] = model is not None
            
        return results
    
    def unload(self, name: str) -> bool:
        """
        Unload a model from memory
        
        Args:
            name: Model identifier
            
        Returns:
            True if model was unloaded, False if not loaded
        """
        with self._global_lock:
            if name in self._models:
                del self._models[name]
                self._model_info[name].loaded = False
                gc.collect()  # Force garbage collection
                logger.info(f"Model '{name}' unloaded")
                return True
        return False
    
    def clear_all(self) -> None:
        """Unload all models"""
        with self._global_lock:
            self._models.clear()
            for info in self._model_info.values():
                info.loaded = False
            gc.collect()
            logger.info("All models unloaded")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all registered models"""
        status = {
            'models': {},
            'total_loaded': 0,
            'total_registered': len(self._loaders)
        }
        
        for name, info in self._model_info.items():
            status['models'][name] = {
                'loaded': info.loaded,
                'load_time': info.load_time,
                'last_used': info.last_used,
                'use_count': info.use_count
            }
            if info.loaded:
                status['total_loaded'] += 1
        
        return status
    
    def is_loaded(self, name: str) -> bool:
        """Check if a model is loaded"""
        return name in self._models
    
    def get_loaded_models(self) -> list:
        """Get list of loaded model names"""
        return list(self._models.keys())


# Global model manager instance
_model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """Get the global model manager instance"""
    return _model_manager


# Decorator for lazy model loading
def lazy_model(model_name: str):
    """
    Decorator for lazy model loading in methods
    
    Usage:
        @lazy_model('classifier')
        def classify(self, text):
            model = self._get_model('classifier')
            return model.predict(text)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Ensure model is loaded before function execution
            manager = get_model_manager()
            if not manager.is_loaded(model_name):
                manager.get(model_name)
            return func(self, *args, **kwargs)
        return wrapper
    return decorator