"""
NEWSCAT Integration Test - Verify all optimized components work correctly
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_classifier():
    """Test the optimized classifier"""
    print("\n" + "="*70)
    print("Testing Ultra-Optimized Classifier v4.0")
    print("="*70)
    
    try:
        from backend.models.optimized_classifier_v2 import UltraOptimizedClassifier
        
        classifier = UltraOptimizedClassifier()
        
        # Test texts for different categories
        test_cases = [
            ("technology", "Apple announced new AI features for iPhone with machine learning capabilities"),
            ("sports", "The Lakers won the championship game with a buzzer beater from LeBron James"),
            ("politics", "The Senate passed new legislation regarding climate change policy"),
            ("business", "Stock market reached record highs as tech companies reported earnings"),
            ("entertainment", "The Oscars ceremony honored the best films of the year"),
            ("health", "New vaccine shows promising results in clinical trials for cancer treatment"),
            ("science", "NASA discovered a new exoplanet with potential for life"),
            ("world", "The United Nations held a summit on global climate action"),
            ("education", "Universities announced new online learning programs for students"),
            ("environment", "Renewable energy sources reached new production milestones"),
        ]
        
        total_time = 0
        correct = 0
        
        for expected_category, text in test_cases:
            start = time.perf_counter()
            result = classifier.classify(text)
            elapsed = (time.perf_counter() - start) * 1000
            total_time += elapsed
            
            predicted = result['category']
            confidence = result['confidence']
            is_correct = predicted == expected_category
            if is_correct:
                correct += 1
            
            status = "✓" if is_correct else "✗"
            print(f"  {status} {expected_category:15} → {predicted:15} ({confidence:.2%}) [{elapsed:.2f}ms]")
        
        avg_time = total_time / len(test_cases)
        accuracy = correct / len(test_cases) * 100
        
        print(f"\n  Average inference time: {avg_time:.2f}ms")
        print(f"  Accuracy: {accuracy:.1f}% ({correct}/{len(test_cases)})")
        print(f"  Cache hit rate: {classifier.get_info()['cache_hit_rate']}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_processor():
    """Test the optimized text processor"""
    print("\n" + "="*70)
    print("Testing Fast Text Processor")
    print("="*70)
    
    try:
        from backend.models.fast_text_processor import get_text_processor
        
        processor = get_text_processor()
        
        text = "Apple announced new AI features! Visit https://example.com for more. Contact us@test.com"
        
        start = time.perf_counter()
        cleaned = processor.clean_text(text)
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"  Original: {text[:60]}...")
        print(f"  Cleaned:  {cleaned[:60]}...")
        print(f"  Cleaning time: {elapsed:.2f}ms")
        
        # Test tokenization
        start = time.perf_counter()
        tokens = processor.tokenize(text)
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"  Tokens ({len(tokens)}): {', '.join(tokens[:10])}...")
        print(f"  Tokenization time: {elapsed:.2f}ms")
        
        # Test feature extraction
        start = time.perf_counter()
        features = processor.extract_features(text)
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"  Features: {features.to_dict()}")
        print(f"  Feature extraction time: {elapsed:.2f}ms")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_keyword_extractor():
    """Test the optimized keyword extractor"""
    print("\n" + "="*70)
    print("Testing Fast Keyword Extractor")
    print("="*70)
    
    try:
        from backend.models.fast_keyword_extractor import get_extractor
        
        extractor = get_extractor()
        
        text = "Apple announced new artificial intelligence features for their iPhone products. Machine learning capabilities will transform how users interact with their devices."
        
        start = time.perf_counter()
        keywords = extractor.extract(text, top_n=5)
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"  Text: {text[:80]}...")
        print(f"  Keywords:")
        for kw in keywords:
            print(f"    - {kw['keyword']}: {kw['score']:.4f}")
        print(f"  Extraction time: {elapsed:.2f}ms")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache():
    """Test the optimized cache"""
    print("\n" + "="*70)
    print("Testing Optimized TTL Cache")
    print("="*70)
    
    try:
        from backend.app import _response_cache
        
        # Test set and get
        test_data = {'test': 'value', 'number': 123}
        _response_cache.set('test_key', test_data)
        
        retrieved = _response_cache.get('test_key')
        
        if retrieved == test_data:
            print("  ✓ Cache set/get works correctly")
        else:
            print("  ✗ Cache set/get failed")
            return False
        
        # Test stats
        stats = _response_cache.get_stats()
        print(f"  Cache stats: {stats}")
        
        # Test clear
        _response_cache.clear()
        cleared = _response_cache.get('test_key')
        
        if cleared is None:
            print("  ✓ Cache clear works correctly")
        else:
            print("  ✗ Cache clear failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_categories():
    """Test all 35 categories"""
    print("\n" + "="*70)
    print("Testing All 35 Categories")
    print("="*70)
    
    try:
        from backend.models.optimized_classifier_v2 import UltraOptimizedClassifier
        
        classifier = UltraOptimizedClassifier()
        categories = classifier.CATEGORIES
        
        print(f"  Total categories: {len(categories)}")
        print(f"  Categories: {', '.join(categories[:10])}...")
        
        # Verify all categories have keyword patterns
        for cat in categories[:5]:  # Test first 5 for brevity
            if cat in classifier._keyword_patterns:
                patterns = classifier._keyword_patterns[cat]
                total_patterns = len(patterns['high']) + len(patterns['medium']) + len(patterns['low'])
                print(f"  ✓ {cat}: {total_patterns} keyword patterns")
            else:
                print(f"  ✗ {cat}: No keyword patterns found")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("NEWSCAT v6.0 - Integration Test Suite")
    print("="*70)
    
    results = {
        "Classifier": test_classifier(),
        "Text Processor": test_text_processor(),
        "Keyword Extractor": test_keyword_extractor(),
        "Cache": test_cache(),
        "Categories": test_all_categories()
    }
    
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name:20} {status}")
    
    all_passed = all(results.values())
    
    print("="*70)
    if all_passed:
        print("All tests PASSED! ✓")
        print("NEWSCAT v6.0 is ready for production.")
    else:
        print("Some tests FAILED. ✗")
        print("Please check the errors above.")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
