"""
Result Merger Module for NEWSCAT
================================
Merges classification results from all model types into unified output.

Features:
- Combines results from text, audio, image, and video models
- Calculates weighted confidence scores
- Handles partial failures gracefully
- Returns unified comprehensive output

Author: NEWSCAT Team
Version: 1.0.0
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import threading

logger = logging.getLogger(__name__)

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ModelResult:
    """Result from a single model"""
    model_type: str  # text, audio, image, video
    success: bool
    categories: List[Dict[str, Any]] = field(default_factory=list)
    primary_category: str = ""
    confidence: float = 0.0
    error_message: str = ""
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'model_type': self.model_type,
            'success': self.success,
            'categories': self.categories,
            'primary_category': self.primary_category,
            'confidence': self.confidence,
            'error_message': self.error_message,
            'processing_time': self.processing_time,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }


@dataclass
class MergedResult:
    """Merged result from all models"""
    success: bool
    primary_category: str = ""
    confidence: float = 0.0
    all_categories: List[Dict[str, Any]] = field(default_factory=list)
    model_results: Dict[str, ModelResult] = field(default_factory=dict)
    partial_failures: List[str] = field(default_factory=list)
    total_processing_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'success': self.success,
            'primary_category': self.primary_category,
            'confidence': self.confidence,
            'all_categories': self.all_categories,
            'model_results': {
                k: v.to_dict() for k, v in self.model_results.items()
            },
            'partial_failures': self.partial_failures,
            'total_processing_time': self.total_processing_time,
            'timestamp': self.timestamp,
            'warnings': self.warnings
        }


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_WEIGHTS = {
    'text': 0.4,      # Text classification is primary
    'audio': 0.2,     # Audio transcription
    'image': 0.2,     # Image analysis  
    'video': 0.2      # Video analysis
}

# Category priority mapping (higher priority = more likely to be final)
CATEGORY_PRIORITY = {
    'technology': 10,
    'politics': 9,
    'business': 8,
    'sports': 7,
    'entertainment': 6,
    'health': 5,
    'science': 4,
    'world': 3,
    'local': 2,
    'other': 1
}

# =============================================================================
# RESULT MERGER CLASS
# =============================================================================

class ResultMerger:
    """
    Merges classification results from multiple models into unified output.
    
    Features:
    - Weighted confidence scoring
    - Graceful handling of partial failures
    - Category aggregation and deduplication
    - Processing time tracking
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the result merger.
        
        Args:
            weights: Optional custom weights for each model type
        """
        self.weights = weights or DEFAULT_WEIGHTS
        self._lock = threading.Lock()
        
        # Validate weights
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.001:
            logger.warning(f"Weights sum to {total_weight}, normalizing to 1.0")
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        logger.info(f"ResultMerger initialized with weights: {self.weights}")
    
    def merge_results(
        self,
        model_results: Dict[str, ModelResult],
        start_time: Optional[float] = None
    ) -> MergedResult:
        """
        Merge results from all models.
        
        Args:
            model_results: Dictionary of model results keyed by model type
            start_time: Optional start time for total processing time calculation
            
        Returns:
            Merged result
        """
        if start_time is None:
            start_time = time.time()
        
        with self._lock:
            successful_results = {}
            partial_failures = []
            warnings = []
            
            # Separate successful and failed results
            for model_type, result in model_results.items():
                if result.success:
                    successful_results[model_type] = result
                else:
                    partial_failures.append(model_type)
                    warnings.append(f"Model '{model_type}' failed: {result.error_message}")
            
            # Calculate merged confidence
            merged_confidence = self._calculate_merged_confidence(successful_results)
            
            # Get primary category
            primary_category = self._determine_primary_category(successful_results)
            
            # Aggregate all categories
            all_categories = self._aggregate_categories(successful_results)
            
            # Determine overall success
            # Success if at least text (primary) model succeeded, or any model succeeded
            overall_success = (
                'text' in successful_results or 
                len(successful_results) > 0
            )
            
            # Add warning if text model failed
            if 'text' not in successful_results and len(successful_results) > 0:
                warnings.append("Text classification failed - results based on other models")
            
            total_processing_time = time.time() - start_time
            
            return MergedResult(
                success=overall_success,
                primary_category=primary_category,
                confidence=merged_confidence,
                all_categories=all_categories,
                model_results=model_results,
                partial_failures=partial_failures,
                total_processing_time=total_processing_time,
                warnings=warnings
            )
    
    def _calculate_merged_confidence(
        self,
        successful_results: Dict[str, ModelResult]
    ) -> float:
        """Calculate weighted merged confidence score"""
        if not successful_results:
            return 0.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for model_type, result in successful_results.items():
            weight = self.weights.get(model_type, 0.25)  # Default weight
            weighted_sum += result.confidence * weight
            total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        return 0.0
    
    def _determine_primary_category(
        self,
        successful_results: Dict[str, ModelResult]
    ) -> str:
        """Determine the primary category from all results"""
        if not successful_results:
            return "unknown"
        
        # If text model exists and has valid (not unknown) category, use it
        if 'text' in successful_results:
            text_result = successful_results['text']
            if text_result.primary_category and text_result.primary_category != 'unknown':
                return text_result.primary_category
        
        # Otherwise, find the best category from all successful results
        category_scores: Dict[str, float] = {}
        
        for model_type, result in successful_results.items():
            weight = self.weights.get(model_type, 0.25)
            
            for cat in result.categories:
                cat_name = cat.get('category', '')
                if not cat_name or cat_name == 'unknown':
                    continue
                
                cat_confidence = cat.get('confidence', 0.0)
                
                # Weight the confidence by model weight and category priority
                priority = CATEGORY_PRIORITY.get(cat_name.lower(), 0)
                score = cat_confidence * weight * (1 + priority / 10)
                
                if cat_name in category_scores:
                    category_scores[cat_name] += score
                else:
                    category_scores[cat_name] = score
        
        if not category_scores:
            return "unknown"
        
        # Return category with highest score
        return max(category_scores.items(), key=lambda x: x[1])[0]
    
    def _aggregate_categories(
        self,
        successful_results: Dict[str, ModelResult]
    ) -> List[Dict[str, Any]]:
        """Aggregate categories from all successful results"""
        aggregated: Dict[str, Dict[str, Any]] = {}
        
        for model_type, result in successful_results.items():
            weight = self.weights.get(model_type, 0.25)
            
            for cat in result.categories:
                cat_name = cat.get('category', '')
                if not cat_name:
                    continue
                
                cat_confidence = cat.get('confidence', 0.0)
                
                if cat_name in aggregated:
                    # Update existing category with weighted average
                    existing = aggregated[cat_name]
                    new_count = existing['count'] + 1
                    new_confidence = (
                        (existing['confidence'] * existing['count'] + cat_confidence * weight)
                        / new_count
                    )
                    existing['confidence'] = new_confidence
                    existing['count'] = new_count
                    existing['sources'].append(model_type)
                else:
                    # Create new category entry
                    aggregated[cat_name] = {
                        'category': cat_name,
                        'confidence': cat_confidence,
                        'count': 1,
                        'sources': [model_type]
                    }
        
        # Convert to list and sort by confidence
        result_list = list(aggregated.values())
        result_list.sort(key=lambda x: x['confidence'], reverse=True)
        
        return result_list
    
    def create_model_result(
        self,
        model_type: str,
        categories: List[Dict[str, Any]],
        confidence: float,
        processing_time: float,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ModelResult:
        """
        Create a ModelResult from classification data.
        
        Args:
            model_type: Type of model (text, audio, image, video)
            categories: List of category predictions
            confidence: Overall confidence score
            processing_time: Time taken to process
            error: Optional error message
            metadata: Optional metadata
            
        Returns:
            ModelResult instance
        """
        success = error is None
        
        # Determine primary category
        primary = ""
        if categories:
            # Sort by confidence and get highest
            sorted_cats = sorted(categories, key=lambda x: x.get('confidence', 0), reverse=True)
            primary = sorted_cats[0].get('category', '') if sorted_cats else ""
        
        return ModelResult(
            model_type=model_type,
            success=success,
            categories=categories,
            primary_category=primary,
            confidence=confidence if success else 0.0,
            error_message=error or "",
            processing_time=processing_time,
            metadata=metadata or {}
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def merge_results(
    model_results: Dict[str, ModelResult],
    weights: Optional[Dict[str, float]] = None
) -> MergedResult:
    """
    Utility function to merge results.
    
    Args:
        model_results: Dictionary of model results
        weights: Optional custom weights
        
    Returns:
        Merged result
    """
    merger = ResultMerger(weights)
    return merger.merge_results(model_results)


def create_empty_result() -> MergedResult:
    """Create an empty result for error cases"""
    return MergedResult(
        success=False,
        error_message="No results available"
    )
