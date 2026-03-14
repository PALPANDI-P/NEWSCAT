"""
Train Models Script

This script loads the dataset created by `fetch_training_data.py`,
instantiates the trainable classifiers, trains them on the data,
and saves the compiled models to disk.
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TrainModels')

# Ensure we can import from backend
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(project_root)) # Add e:\NEWSCAT to path

from backend.config import Config
from backend.models.simple_classifier import SimpleNewsClassifier
from backend.models.optimized_classifier import OptimizedEnsembleClassifier

def load_dataset(dataset_path: str) -> tuple:
    """Load and parse the JSON dataset."""
    logger.info(f"Loading dataset from {dataset_path}")
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at {dataset_path}")
        return [], []
        
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        texts = []
        labels = []
        
        for item in data:
            if 'text' in item and 'label' in item:
                texts.append(item['text'])
                labels.append(item['label'])
                
        logger.info(f"Loaded {len(texts)} samples")
        
        # Display distribution
        counts = Counter(labels)
        logger.info("Category distribution:")
        for cat, val in counts.most_common():
            logger.info(f"  {cat}: {val}")
            
        return texts, labels
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return [], []

def train_and_save_models(texts: list, labels: list):
    """Train both classifiers on the provided data."""
    if not texts or not labels:
        logger.error("No data provided for training.")
        return
        
    # Find active categories in our dataset
    active_categories = sorted(list(set(labels)))
    logger.info(f"Training on {len(active_categories)} unique categories")
    
    # 1. Train SimpleNewsClassifier
    logger.info("--- Training SimpleNewsClassifier ---")
    simple_model = SimpleNewsClassifier()
    start_time = time.time()
    success = simple_model.train(texts, labels)
    simple_time = time.time() - start_time
    
    if success:
        logger.info(f"SimpleNewsClassifier trained successfully in {simple_time:.2f}s (Accuracy: {simple_model.accuracy:.2%})")
    else:
        logger.error("SimpleNewsClassifier training failed")
        
    # 2. Train OptimizedEnsembleClassifier
    logger.info("--- Training OptimizedEnsembleClassifier ---")
    optimized_model = OptimizedEnsembleClassifier()
    start_time = time.time()
    
    try:
        # Optimized requires a larger dataset. Ensure we have at least 10 samples total
        if len(texts) < 10:
            logger.warning("Dataset too small for ensemble classifier. Using duplication to meet minimums.")
            texts = texts * (10 // len(texts) + 1)
            labels = labels * (10 // len(labels) + 1)
            
        results = optimized_model.train(texts, labels)
        opt_time = time.time() - start_time
        
        logger.info(f"OptimizedEnsembleClassifier trained successfully in {opt_time:.2f}s")
        logger.info(f"  Accuracy: {results.get('accuracy', 0):.2%}")
        logger.info(f"  Feature Count: {results.get('feature_count', 0)}")
        
    except Exception as e:
        logger.error(f"OptimizedEnsembleClassifier training failed: {e}")

if __name__ == "__main__":
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = project_root / 'data' / 'training'
    dataset_file = data_dir / 'online_dataset.json'
    
    logger.info("Starting model training pipeline")
    
    if not dataset_file.exists():
        logger.error(f"Please run fetch_training_data.py first to generate {dataset_file}")
        sys.exit(1)
        
    texts, labels = load_dataset(str(dataset_file))
    train_and_save_models(texts, labels)
    
    logger.info("Pipeline complete")
