"""
NEWSCAT Model Workers
=====================
Worker modules for parallel processing of different model types.
Each worker runs in its own subprocess for independent execution.

Workers:
- text_worker: Text classification
- audio_worker: Audio processing and classification
- image_worker: Image processing and classification  
- video_worker: Video processing and classification

Author: NEWSCAT Team
Version: 1.0.0
"""

from backend.models.workers.text_worker import process_text
from backend.models.workers.audio_worker import process_audio
from backend.models.workers.image_worker import process_image
from backend.models.workers.video_worker import process_video

__all__ = [
    'process_text',
    'process_audio', 
    'process_image',
    'process_video'
]
