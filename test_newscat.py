#!/usr/bin/env python
"""
NEWSCAT - Comprehensive Test Suite
Verifies all components work correctly before production deployment
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all modules can be imported"""
    print("\n" + "="*60)
    print("TEST 1: Module Imports")
    print("="*60)
    
    try:
        from backend.config import DevelopmentConfig, Config
        print("[OK] Config module loaded")
    except Exception as e:
        print(f"[ERROR] Config: {e}")
        return False
    
    try:
        from backend.models.base_classifier import BaseNewsClassifier
        print("[OK] BaseNewsClassifier loaded")
    except Exception as e:
        print(f"[ERROR] BaseNewsClassifier: {e}")
        return False
    
    try:
        from backend.models.text_processor import TextProcessor
        print("[OK] TextProcessor loaded")
    except Exception as e:
        print(f"[ERROR] TextProcessor: {e}")
        return False
    
    try:
        from backend.models.simple_classifier import SimpleNewsClassifier
        print("[OK] SimpleNewsClassifier loaded")
    except Exception as e:
        print(f"[ERROR] SimpleNewsClassifier: {e}")
        return False
    
    try:
        from backend.models.enhanced_classifier import EnhancedNewsClassifier
        print("[OK] EnhancedNewsClassifier loaded")
    except Exception as e:
        print(f"[ERROR] EnhancedNewsClassifier: {e}")
        return False
    
    try:
        from backend.models.keyword_extractor import KeywordExtractor
        print("[OK] KeywordExtractor loaded")
    except Exception as e:
        print(f"[ERROR] KeywordExtractor: {e}")
        return False
    
    try:
        from backend.app import app, classifiers
        print("[OK] Flask app loaded")
    except Exception as e:
        print(f"[ERROR] Flask app: {e}")
        return False
    
    return True


def test_classifiers():
    """Test classifier initialization"""
    print("\n" + "="*60)
    print("TEST 2: Classifier Initialization")
    print("="*60)
    
    from backend.app import classifiers
    
    if classifiers['simple'] is None:
        print("[ERROR] SimpleClassifier failed to initialize")
        return False
    print(f"[OK] SimpleClassifier: {classifiers['simple'].name}")
    
    if classifiers['enhanced'] is None:
        print("[ERROR] EnhancedClassifier failed to initialize")
        return False
    print(f"[OK] EnhancedClassifier: {classifiers['enhanced'].name}")
    
    if classifiers['keyword_extractor'] is None:
        print("[ERROR] KeywordExtractor failed to initialize")
        return False
    print("[OK] KeywordExtractor initialized")
    
    return True


def test_classification():
    """Test text classification"""
    print("\n" + "="*60)
    print("TEST 3: Text Classification")
    print("="*60)
    
    from backend.app import classifiers
    
    test_cases = [
        ("Apple released new iPhone with advanced AI features", "technology"),
        ("Manchester United won the Premier League match", "sports"),
        ("Congress passed new healthcare bill today", "politics"),
        ("Stock market reached record highs this quarter", "business"),
    ]
    
    classifier = classifiers['simple']
    for text, expected_category in test_cases:
        try:
            result = classifier.classify(text)
            category = result['category']
            confidence = result['confidence']
            print(f"[OK] '{text[:40]}...' -> {category} ({confidence:.1%})")
        except Exception as e:
            print(f"[ERROR] Classification failed: {e}")
            return False
    
    return True


def test_api():
    """Test Flask API endpoints"""
    print("\n" + "="*60)
    print("TEST 4: API Endpoints")
    print("="*60)
    
    from backend.app import app
    
    client = app.test_client()
    
    # Test health endpoint
    response = client.get('/api/health')
    if response.status_code != 200:
        print(f"[ERROR] /api/health returned {response.status_code}")
        return False
    data = response.get_json()
    if data['status'] != 'healthy':
        print(f"[ERROR] Health status: {data['status']}")
        return False
    print(f"[OK] /api/health - Status: {data['status']}")
    
    # Test categories endpoint
    response = client.get('/api/categories')
    if response.status_code != 200:
        print(f"[ERROR] /api/categories returned {response.status_code}")
        return False
    data = response.get_json()
    if len(data['categories']) != 10:
        print(f"[ERROR] Expected 10 categories, got {len(data['categories'])}")
        return False
    print(f"[OK] /api/categories - {data['count']} categories")
    
    # Test classify endpoint
    response = client.post('/api/classify', json={
        "text": "Apple released new iPhone with advanced AI features",
        "enhanced": False
    })
    if response.status_code != 200:
        print(f"[ERROR] /api/classify returned {response.status_code}")
        return False
    data = response.get_json()
    if 'category' not in data:
        print(f"[ERROR] Missing 'category' in response")
        return False
    print(f"[OK] /api/classify - Category: {data['category']}, Confidence: {data['confidence']:.1%}")
    
    # Test model info endpoint
    response = client.get('/api/model/info?enhanced=false')
    if response.status_code != 200:
        print(f"[ERROR] /api/model/info returned {response.status_code}")
        return False
    data = response.get_json()
    print(f"[OK] /api/model/info - Model: {data['name']}, Version: {data['version']}")
    
    return True


def test_text_processing():
    """Test text preprocessing"""
    print("\n" + "="*60)
    print("TEST 5: Text Processing")
    print("="*60)
    
    from backend.models.text_processor import TextProcessor
    
    processor = TextProcessor(use_advanced=False)
    
    test_text = "Visit https://example.com! Email: test@example.com. <html>Remove tags</html> 123numbers"
    
    # Test clean_text
    cleaned = processor.clean_text(test_text)
    if len(cleaned) == 0:
        print("[ERROR] clean_text returned empty string")
        return False
    print(f"[OK] clean_text - Input: {len(test_text)} chars, Output: {len(cleaned)} chars")
    
    # Test tokenize
    tokens = processor.tokenize(test_text)
    if len(tokens) == 0:
        print("[ERROR] tokenize returned empty list")
        return False
    print(f"[OK] tokenize - {len(tokens)} tokens extracted")
    
    # Test preprocess_pipeline
    processed = processor.preprocess_pipeline(test_text)
    if len(processed) == 0:
        print("[ERROR] preprocess_pipeline returned empty string")
        return False
    print(f"[OK] preprocess_pipeline - Final: '{processed[:50]}...'")
    
    return True


def test_configuration():
    """Test configuration loading"""
    print("\n" + "="*60)
    print("TEST 6: Configuration")
    print("="*60)
    
    from backend.app import app
    
    required_config = [
        'MIN_TEXT_LENGTH',
        'MAX_TEXT_LENGTH',
        'TFIDF_MAX_FEATURES',
        'NGRAM_RANGE',
        'CATEGORIES',
        'HOST',
        'PORT'
    ]
    
    for key in required_config:
        if key not in app.config:
            print(f"[ERROR] Missing configuration: {key}")
            return False
        print(f"[OK] {key}: {app.config[key]}")
    
    return True


def main():
    """Run all tests"""
    print("\n")
    print("=" * 60)
    print("NEWSCAT - Comprehensive Test Suite".center(60))
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Classifier Initialization", test_classifiers),
        ("Text Classification", test_classification),
        ("API Endpoints", test_api),
        ("Text Processing", test_text_processing),
        ("Configuration", test_configuration),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[CRITICAL ERROR] {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "[+]" if result else "[-]"
        print(f"{symbol} {test_name}: {status}")
    
    print("="*60)
    print(f"Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed! System is ready for production.")
        print("\nTo start the server, run:")
        print("  PowerShell: .\\run.ps1")
        print("  Direct: python backend/app.py")
        print("\nThen open: http://localhost:5000")
        return 0
    else:
        print(f"\n[FAILED] {total - passed} test(s) failed. Please fix the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
