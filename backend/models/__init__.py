"""
NEWSCAT v10.0 - Quantum AI Models Package
==========================================
Neural Classification System with Advanced ML

Modules:
- lightning_classifier: QuantumClassifier with transformer embeddings
- audio_processor: NeuralAudioProcessor with Whisper large-v3
- image_processor: VisionProcessor with CLIP embeddings
- video_processor: CinematicProcessor with scene analysis
"""

# Import key classes for easier access
from .base_classifier import BaseNewsClassifier

# v10.0 Quantum AI Models
from .lightning_classifier import (
    QuantumClassifier,
    LightningClassifier,
    get_classifier,
    classify_text
)
from .audio_processor import (
    NeuralAudioProcessor,
    AudioProcessor,
    get_audio_processor
)
from .image_processor import (
    VisionProcessor,
    ImageProcessor,
    get_image_processor
)
from .video_processor import (
    CinematicProcessor,
    VideoProcessor,
    get_video_processor
)

__all__ = [
    # Base classes
    'BaseNewsClassifier',
    # v10.0 Quantum AI Models
    'QuantumClassifier',
    'LightningClassifier',
    'NeuralAudioProcessor',
    'AudioProcessor',
    'VisionProcessor',
    'ImageProcessor',
    'CinematicProcessor',
    'VideoProcessor',
    # Helper functions
    'get_classifier',
    'get_audio_processor',
    'get_image_processor',
    'get_video_processor',
    'classify_text'
]
