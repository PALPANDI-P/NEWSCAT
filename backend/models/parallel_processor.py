"""
Parallel Processor Controller for NEWSCAT
==========================================
Main controller that manages parallel execution of all model types.
Uses ProcessPoolExecutor for concurrent subprocess execution.

Features:
- Parallel execution of text, audio, image, video models
- Independent subprocess for each model type
- Global timeout management (5 seconds)
- Graceful degradation on partial failures
- Result merging and confidence calculation

Author: NEWSCAT Team
Version: 1.0.0
"""

import os
import sys
import time
import logging
import json
import traceback
from typing import Dict, Any, Optional, List
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
import threading

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import modules
from backend.models.result_merger import ResultMerger, ModelResult, MergedResult

# Import workers
from backend.models.workers import (
    process_text,
    process_audio,
    process_image,
    process_video
)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    'global_timeout': 5.0,  # Global timeout in seconds
    'model_timeout': 4.0,  # Individual model timeout
    'max_workers': 4,  # Maximum parallel workers
    'weights': {
        'text': 0.4,
        'audio': 0.2,
        'image': 0.2,
        'video': 0.2
    }
}


class ParallelProcessor:
    """
    Main parallel processor controller.
    
    Manages concurrent execution of all model types (text, audio, image, video)
    using ProcessPoolExecutor for true parallel processing.
    
    Features:
    - Spawns independent subprocess for each model type
    - Collects and merges results from all models
    - Global timeout management
    - Graceful degradation on failures
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the parallel processor.
        
        Args:
            config: Optional configuration override
        """
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self._result_merger = ResultMerger(weights=self.config.get('weights'))
        self._lock = threading.Lock()
        
        # Model functions mapping
        self._model_functions = {
            'text': process_text,
            'audio': process_audio,
            'image': process_image,
            'video': process_video
        }
        
        logger.info(f"ParallelProcessor initialized with config: {self.config}")
    
    def process(
        self,
        text: Optional[str] = None,
        audio_path: Optional[str] = None,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        audio_data: Optional[bytes] = None,
        image_data: Optional[bytes] = None,
        video_data: Optional[bytes] = None,
        audio_base64: Optional[str] = None,
        image_base64: Optional[str] = None,
        video_base64: Optional[str] = None,
        timeout: Optional[float] = None,
        models: Optional[List[str]] = None
    ) -> MergedResult:
        """
        Process input data using all model types in parallel.
        
        Args:
            text: Text input for text classification
            audio_path: Path to audio file
            image_path: Path to image file
            video_path: Path to video file
            audio_data: Raw audio bytes
            image_data: Raw image bytes
            video_data: Raw video bytes
            audio_base64: Base64 encoded audio
            image_base64: Base64 encoded image
            video_base64: Base64 encoded video
            timeout: Optional timeout override (default: 5 seconds)
            models: Optional list of models to run (default: all)
            
        Returns:
            MergedResult containing all classification results
        """
        start_time = time.time()
        
        # Determine which models to run
        if models is None:
            models = ['text', 'audio', 'image', 'video']
        
        # Prepare input data for each model
        model_inputs = {}
        
        if 'text' in models and text:
            model_inputs['text'] = {'text': text}
        
        if 'audio' in models:
            if audio_path:
                model_inputs['audio'] = {'audio_path': audio_path}
            elif audio_data:
                model_inputs['audio'] = {'audio_data': audio_data}
            elif audio_base64:
                model_inputs['audio'] = {'audio_base64': audio_base64}
            elif text:  # Fallback: use text for audio classification
                model_inputs['audio'] = {'text': text}
        
        if 'image' in models:
            if image_path:
                model_inputs['image'] = {'image_path': image_path}
            elif image_data:
                model_inputs['image'] = {'image_data': image_data}
            elif image_base64:
                model_inputs['image'] = {'image_base64': image_base64}
            elif text:  # Fallback: use text for image classification
                model_inputs['image'] = {'text': text}
        
        if 'video' in models:
            if video_path:
                model_inputs['video'] = {'video_path': video_path}
            elif video_data:
                model_inputs['video'] = {'video_data': video_data}
            elif video_base64:
                model_inputs['video'] = {'video_base64': video_base64}
            elif text:  # Fallback: use text for video classification
                model_inputs['video'] = {'text': text}
        
        # Execute models in parallel
        results = self._execute_parallel(
            model_inputs,
            timeout or self.config.get('global_timeout', 5.0)
        )
        
        # Merge results
        merged_result = self._result_merger.merge_results(results, start_time)
        
        # Log results
        logger.info(
            f"Parallel processing complete: {len(results)} models, "
            f"success={merged_result.success}, "
            f"category={merged_result.primary_category}, "
            f"confidence={merged_result.confidence:.2f}, "
            f"time={merged_result.total_processing_time:.2f}s"
        )
        
        return merged_result
    
    def _execute_parallel(
        self,
        model_inputs: Dict[str, Dict[str, Any]],
        timeout: float
    ) -> Dict[str, ModelResult]:
        """
        Execute all models in parallel using ProcessPoolExecutor.
        
        Args:
            model_inputs: Dictionary of input data for each model
            timeout: Global timeout for all operations
            
        Returns:
            Dictionary of ModelResult for each model
        """
        results: Dict[str, ModelResult] = {}
        
        if not model_inputs:
            logger.warning("No model inputs provided")
            return results
        
        # Determine effective timeout per model
        model_timeout = min(
            self.config.get('model_timeout', 4.0),
            timeout / max(len(model_inputs), 1)
        )
        
        # Use ProcessPoolExecutor for true parallel execution
        with ProcessPoolExecutor(max_workers=self.config.get('max_workers', 4)) as executor:
            # Submit all tasks
            future_to_model = {}
            
            for model_type, input_data in model_inputs.items():
                model_func = self._model_functions.get(model_type)
                if model_func:
                    future = executor.submit(model_func, input_data)
                    future_to_model[future] = model_type
                    logger.debug(f"Submitted {model_type} model for processing")
            
            # Collect results with timeout
            remaining_timeout = timeout
            
            for future in as_completed(future_to_model, timeout=remaining_timeout):
                model_type = future_to_model[future]
                model_start = time.time()
                
                try:
                    # Get result with timeout
                    result_data = future.result(timeout=model_timeout)
                    
                    # Convert to ModelResult
                    model_result = self._create_model_result(model_type, result_data)
                    results[model_type] = model_result
                    
                    logger.debug(
                        f"{model_type} model completed: "
                        f"success={model_result.success}, "
                        f"time={model_result.processing_time:.2f}s"
                    )
                    
                except TimeoutError:
                    logger.warning(f"{model_type} model timed out after {model_timeout}s")
                    results[model_type] = self._create_error_result(
                        model_type,
                        f"Timeout after {model_timeout}s",
                        time.time() - model_start
                    )
                    
                except Exception as e:
                    logger.error(f"{model_type} model failed: {e}")
                    logger.error(traceback.format_exc())
                    results[model_type] = self._create_error_result(
                        model_type,
                        str(e),
                        time.time() - model_start
                    )
        
        return results
    
    def _create_model_result(
        self,
        model_type: str,
        result_data: Dict[str, Any]
    ) -> ModelResult:
        """Create a ModelResult from worker result data"""
        return ModelResult(
            model_type=model_type,
            success=result_data.get('success', False),
            categories=result_data.get('categories', []),
            primary_category=result_data.get('primary_category', 'unknown'),
            confidence=result_data.get('confidence', 0.0),
            error_message=result_data.get('error', '') or result_data.get('error_message', ''),
            processing_time=result_data.get('processing_time', 0.0),
            metadata=result_data.get('metadata', {})
        )
    
    def _create_error_result(
        self,
        model_type: str,
        error_message: str,
        processing_time: float
    ) -> ModelResult:
        """Create an error ModelResult"""
        return ModelResult(
            model_type=model_type,
            success=False,
            categories=[],
            primary_category='unknown',
            confidence=0.0,
            error_message=error_message,
            processing_time=processing_time,
            metadata={}
        )
    
    def process_text_only(self, text: str) -> MergedResult:
        """Process text only (convenience method)"""
        return self.process(text=text, models=['text'])
    
    def process_all(self, data: Dict[str, Any]) -> MergedResult:
        """
        Process mixed input data.
        
        Args:
            data: Dictionary containing any of:
                - text: str
                - audio_path: str
                - image_path: str
                - video_path: str
                - audio_data: bytes
                - image_data: bytes
                - video_data: bytes
                - audio_base64: str
                - image_base64: str
                - video_base64: str
                
        Returns:
            MergedResult
        """
        return self.process(
            text=data.get('text'),
            audio_path=data.get('audio_path'),
            image_path=data.get('image_path'),
            video_path=data.get('video_path'),
            audio_data=data.get('audio_data'),
            image_data=data.get('image_data'),
            video_data=data.get('video_data'),
            audio_base64=data.get('audio_base64'),
            image_base64=data.get('image_base64'),
            video_base64=data.get('video_base64'),
            timeout=data.get('timeout'),
            models=data.get('models')
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Global processor instance
_processor: Optional[ParallelProcessor] = None
_processor_lock = threading.Lock()


def get_processor(config: Optional[Dict[str, Any]] = None) -> ParallelProcessor:
    """
    Get or create the global parallel processor instance.
    
    Args:
        config: Optional configuration override
        
    Returns:
        ParallelProcessor instance
    """
    global _processor
    
    with _processor_lock:
        if _processor is None:
            _processor = ParallelProcessor(config)
    
    return _processor


def process_classification(
    text: Optional[str] = None,
    audio_path: Optional[str] = None,
    image_path: Optional[str] = None,
    video_path: Optional[str] = None,
    timeout: Optional[float] = 5.0,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function for classification.
    
    Args:
        text: Text input
        audio_path: Path to audio file
        image_path: Path to image file
        video_path: Path to video file
        timeout: Timeout in seconds (default: 5)
        config: Optional configuration
        
    Returns:
        Dictionary with classification results
    """
    processor = get_processor(config)
    
    result = processor.process(
        text=text,
        audio_path=audio_path,
        image_path=image_path,
        video_path=video_path,
        timeout=timeout
    )
    
    return result.to_dict()


# =============================================================================
# FLASK INTEGRATION
# =============================================================================

def create_classification_endpoint(app, processor: Optional[ParallelProcessor] = None):
    """
    Create Flask route for parallel classification.
    
    Args:
        app: Flask application instance
        processor: Optional ParallelProcessor instance
    """
    from flask import request, jsonify
    
    processor = processor or get_processor()
    
    @app.route('/api/classify/parallel', methods=['POST'])
    def classify_parallel():
        """Parallel classification endpoint"""
        try:
            data = request.get_json() or {}
            
            # Extract parameters
            text = data.get('text')
            audio_path = data.get('audio_path')
            image_path = data.get('image_path')
            video_path = data.get('video_path')
            audio_base64 = data.get('audio_base64')
            image_base64 = data.get('image_base64')
            video_base64 = data.get('video_base64')
            timeout = data.get('timeout', 5.0)
            models = data.get('models')  # Optional: ['text', 'audio', etc.]
            
            # Process
            result = processor.process(
                text=text,
                audio_path=audio_path,
                image_path=image_path,
                video_path=video_path,
                audio_base64=audio_base64,
                image_base64=image_base64,
                video_base64=video_base64,
                timeout=timeout,
                models=models
            )
            
            return jsonify(result.to_dict())
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': str(e),
                'primary_category': 'unknown',
                'confidence': 0.0
            }), 500
    
    return classify_parallel


# For direct testing
if __name__ == '__main__':
    # Test the parallel processor
    processor = ParallelProcessor()
    
    test_text = """
    Technology stocks surged today as Apple and Google announced new AI features.
    The Nasdaq composite rose 2.5% while the Dow Jones industrial average gained 500 points.
    Analysts predict continued growth in the tech sector.
    """
    
    print("Testing parallel processor with text input...")
    result = processor.process(
        text=test_text,
        models=['text', 'audio', 'image', 'video']
    )
    
    print(json.dumps(result.to_dict(), indent=2))
