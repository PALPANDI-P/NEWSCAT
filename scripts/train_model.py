#!/usr/bin/env python
"""
NEWSCAT Model Training Script
Train and evaluate the ensemble classifier with custom data

Usage:
    python scripts/train_model.py --data data/training/news_samples.json
    python scripts/train_model.py --data data/training/news_samples.json --validate
    python scripts/train_model.py --data data/training/news_samples.json --output models/custom_model.joblib

Performance:
    - Training: O(n * m) where n=samples, m=features
    - Cross-validation: 5-fold by default
    - Expected accuracy: 85-92% with sufficient data
"""

import argparse
import json
import sys
import os
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_training_data(data_path: Path) -> Tuple[List[str], List[str]]:
    """
    Load training data from JSON file
    
    Expected format:
    [
        {"text": "News article text...", "category": "technology"},
        {"text": "Another article...", "category": "sports"},
        ...
    ]
    
    Args:
        data_path: Path to JSON file
        
    Returns:
        Tuple of (texts, labels)
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")
    
    logger.info(f"Loading training data from {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = []
    labels = []
    
    for item in data:
        if 'text' not in item or 'category' not in item:
            logger.warning(f"Skipping invalid item: {item.get('id', 'unknown')}")
            continue
        
        texts.append(item['text'])
        labels.append(item['category'])
    
    logger.info(f"Loaded {len(texts)} samples")
    
    return texts, labels


def validate_data(texts: List[str], labels: List[str]) -> Dict[str, Any]:
    """
    Validate training data and return statistics
    
    Args:
        texts: List of text samples
        labels: List of category labels
        
    Returns:
        Dictionary with validation results
    """
    stats = {
        'total_samples': len(texts),
        'categories': {},
        'text_lengths': {
            'min': float('inf'),
            'max': 0,
            'avg': 0
        },
        'issues': []
    }
    
    # Count categories
    for label in labels:
        stats['categories'][label] = stats['categories'].get(label, 0) + 1
    
    # Check text lengths
    total_length = 0
    for i, text in enumerate(texts):
        length = len(text)
        total_length += length
        
        if length < stats['text_lengths']['min']:
            stats['text_lengths']['min'] = length
        if length > stats['text_lengths']['max']:
            stats['text_lengths']['max'] = length
        
        # Check for issues
        if length < 20:
            stats['issues'].append(f"Sample {i}: Text too short ({length} chars)")
        if length > 10000:
            stats['issues'].append(f"Sample {i}: Text too long ({length} chars)")
    
    stats['text_lengths']['avg'] = total_length / len(texts) if texts else 0
    
    # Check category balance
    category_counts = list(stats['categories'].values())
    if category_counts:
        min_count = min(category_counts)
        max_count = max(category_counts)
        if max_count > min_count * 3:
            stats['issues'].append(
                f"Imbalanced categories: max={max_count}, min={min_count}"
            )
    
    return stats


def print_stats(stats: Dict[str, Any]):
    """Print validation statistics"""
    print("\n" + "="*60)
    print("DATA STATISTICS")
    print("="*60)
    
    print(f"\nTotal Samples: {stats['total_samples']}")
    
    print("\nCategory Distribution:")
    for category, count in sorted(stats['categories'].items()):
        percentage = (count / stats['total_samples']) * 100
        bar = '!' * int(percentage / 2)
        print(f"  {category:15} {count:5} ({percentage:5.1f}%) {bar}")
    
    print(f"\nText Lengths:")
    print(f"  Min: {stats['text_lengths']['min']:.0f} chars")
    print(f"  Max: {stats['text_lengths']['max']:.0f} chars")
    print(f"  Avg: {stats['text_lengths']['avg']:.0f} chars")
    
    if stats['issues']:
        print(f"\nIssues ({len(stats['issues'])}):")
        for issue in stats['issues'][:10]:  # Show first 10
            print(f"  - {issue}")
        if len(stats['issues']) > 10:
            print(f"  ... and {len(stats['issues']) - 10} more")


def train_classifier(
    texts: List[str],
    labels: List[str],
    validate: bool = True,
    output_path: str = None
) -> Dict[str, Any]:
    """
    Train the ensemble classifier
    
    Args:
        texts: Training texts
        labels: Category labels
        validate: Whether to perform cross-validation
        output_path: Optional path to save model
        
    Returns:
        Training results
    """
    from backend.models.ensemble_classifier import EnsembleNewsClassifier
    from backend.config import Config
    
    logger.info("Initializing Ensemble Classifier...")
    
    # Initialize classifier with config
    config = {
        'TFIDF_MAX_FEATURES': Config.TFIDF_MAX_FEATURES,
        'NGRAM_RANGE': Config.NGRAM_RANGE,
        'MIN_TEXT_LENGTH': Config.MIN_TEXT_LENGTH,
        'MAX_TEXT_LENGTH': Config.MAX_TEXT_LENGTH,
        'CATEGORIES': Config.CATEGORIES
    }
    
    classifier = EnsembleNewsClassifier(config=config)
    
    # Train
    logger.info("Starting training...")
    start_time = datetime.now()
    
    results = classifier.train(texts, labels, validate=validate)
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Add training time to results
    results['training_time_seconds'] = training_time
    
    # Save to custom path if specified
    if output_path:
        classifier.save_model(output_path)
        logger.info(f"Model saved to {output_path}")
    
    return results


def print_results(results: Dict[str, Any]):
    """Print training results"""
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    
    print(f"\nAccuracy: {results['accuracy']:.2%}")
    
    if results.get('cv_mean'):
        print(f"Cross-Validation: {results['cv_mean']:.2%} (+/- {results['cv_std']:.2%})")
    
    print(f"\nFeature Count: {results.get('feature_count', 'N/A')}")
    print(f"Training Time: {results.get('training_time_seconds', 0):.2f} seconds")
    
    # Print classification report
    if 'report' in results:
        print("\nClassification Report:")
        print("-" * 60)
        
        report = results['report']
        
        # Header
        print(f"{'Category':15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print("-" * 60)
        
        # Per-category metrics
        for category, metrics in sorted(report.items()):
            if isinstance(metrics, dict) and 'precision' in metrics:
                if category not in ['accuracy', 'macro avg', 'weighted avg']:
                    print(
                        f"{category:15} "
                        f"{metrics['precision']:10.2f} "
                        f"{metrics['recall']:10.2f} "
                        f"{metrics['f1-score']:10.2f} "
                        f"{metrics['support']:10.0f}"
                    )
        
        print("-" * 60)
        
        # Averages
        if 'weighted avg' in report:
            avg = report['weighted avg']
            print(
                f"{'Weighted Avg':15} "
                f"{avg['precision']:10.2f} "
                f"{avg['recall']:10.2f} "
                f"{avg['f1-score']:10.2f} "
                f"{avg['support']:10.0f}"
            )
    
    print("="*60)


def test_classifier(texts: List[str], labels: List[str], sample_size: int = 5):
    """
    Test classifier on sample texts
    
    Args:
        texts: Test texts
        labels: True labels
        sample_size: Number of samples to test
    """
    from backend.models.ensemble_classifier import EnsembleNewsClassifier
    
    logger.info("\nTesting classifier on sample data...")
    
    classifier = EnsembleNewsClassifier()
    
    # Select random samples
    import random
    indices = random.sample(range(len(texts)), min(sample_size, len(texts)))
    
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    correct = 0
    for i in indices:
        text = texts[i]
        true_label = labels[i]
        
        result = classifier.classify(text)
        predicted = result['category']
        confidence = result['confidence']
        
        is_correct = predicted == true_label
        if is_correct:
            correct += 1
        
        status = "!" if is_correct else "?"
        print(f"\n[{status}] Text: {text[:60]}...")
        print(f"    True: {true_label:15} | Predicted: {predicted:15} ({confidence:.1%})")
    
    accuracy = correct / len(indices)
    print(f"\nSample Accuracy: {accuracy:.1%} ({correct}/{len(indices)})")
    print("="*60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Train NEWSCAT ensemble classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python scripts/train_model.py --data data/training/news_samples.json
    
    # Training with cross-validation
    python scripts/train_model.py --data data/training/news_samples.json --validate
    
    # Save to custom path
    python scripts/train_model.py --data data/training/news_samples.json --output models/my_model.joblib
    
    # Test after training
    python scripts/train_model.py --data data/training/news_samples.json --test
        """
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to training data JSON file'
    )
    
    parser.add_argument(
        '--validate', '-v',
        action='store_true',
        help='Perform 5-fold cross-validation'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output path for trained model'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test classifier on sample data after training'
    )
    
    parser.add_argument(
        '--test-size',
        type=int,
        default=5,
        help='Number of samples to test (default: 5)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed output'
    )
    
    args = parser.parse_args()
    
    # Set log level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Load data
    data_path = Path(args.data)
    try:
        texts, labels = load_training_data(data_path)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1
    
    # Validate data
    stats = validate_data(texts, labels)
    
    if not args.quiet:
        print_stats(stats)
    
    # Check minimum requirements
    if len(texts) < 10:
        logger.error("Need at least 10 training samples")
        return 1
    
    if len(set(labels)) < 2:
        logger.error("Need at least 2 different categories")
        return 1
    
    # Train
    try:
        results = train_classifier(
            texts,
            labels,
            validate=args.validate,
            output_path=args.output
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Print results
    if not args.quiet:
        print_results(results)
    
    # Test
    if args.test:
        test_classifier(texts, labels, sample_size=args.test_size)
    
    # Summary
    print("\n" + "="*60)
    if results['accuracy'] >= 0.85:
        print("[SUCCESS] Model trained successfully with good accuracy!")
    elif results['accuracy'] >= 0.70:
        print("[OK] Model trained. Consider adding more training data.")
    else:
        print("[WARNING] Low accuracy. Add more diverse training data.")
    
    print(f"\nFinal Accuracy: {results['accuracy']:.2%}")
    print("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())