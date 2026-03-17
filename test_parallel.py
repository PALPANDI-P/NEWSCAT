#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test script for parallel processing system"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set output encoding
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def test_imports():
    """Test all imports"""
    print("=" * 50)
    print("Testing imports...")
    print("=" * 50)
    
    try:
        from backend.models.parallel_processor import ParallelProcessor
        print("[OK] ParallelProcessor imported")
        
        from backend.models.network_resilience import NetworkResilience
        print("[OK] NetworkResilience imported")
        
        from backend.models.result_merger import ResultMerger, ModelResult
        print("[OK] ResultMerger imported")
        
        from backend.models.workers import process_text, process_audio, process_image, process_video
        print("[OK] Workers imported")
        
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return False


def test_text_worker():
    """Test text worker"""
    print("\n" + "=" * 50)
    print("Testing text worker...")
    print("=" * 50)
    
    try:
        from backend.models.workers import process_text
        result = process_text({'text': 'Technology stocks surged today as Apple announced new AI features'})
        
        print(f"Success: {result.get('success')}")
        print(f"Primary category: {result.get('primary_category')}")
        print(f"Confidence: {result.get('confidence')}")
        
        return result.get('success') == True
    except Exception as e:
        print(f"[FAIL] Text worker failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_worker():
    """Test audio worker"""
    print("\n" + "=" * 50)
    print("Testing audio worker...")
    print("=" * 50)
    
    try:
        from backend.models.workers import process_audio
        result = process_audio({'text': 'The president announced new economic policies'})
        
        print(f"Success: {result.get('success')}")
        print(f"Primary category: {result.get('primary_category')}")
        print(f"Confidence: {result.get('confidence')}")
        
        return result.get('success') == True
    except Exception as e:
        print(f"[FAIL] Audio worker failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_worker():
    """Test image worker"""
    print("\n" + "=" * 50)
    print("Testing image worker...")
    print("=" * 50)
    
    try:
        from backend.models.workers import process_image
        result = process_image({'text': 'Stock market trading floor with investors'})
        
        print(f"Success: {result.get('success')}")
        print(f"Primary category: {result.get('primary_category')}")
        print(f"Confidence: {result.get('confidence')}")
        
        return result.get('success') == True
    except Exception as e:
        print(f"[FAIL] Image worker failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_video_worker():
    """Test video worker"""
    print("\n" + "=" * 50)
    print("Testing video worker...")
    print("=" * 50)
    
    try:
        from backend.models.workers import process_video
        result = process_video({'text': 'Football match highlights showing players scoring'})
        
        print(f"Success: {result.get('success')}")
        print(f"Primary category: {result.get('primary_category')}")
        print(f"Confidence: {result.get('confidence')}")
        
        return result.get('success') == True
    except Exception as e:
        print(f"[FAIL] Video worker failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parallel_processor():
    """Test parallel processor"""
    print("\n" + "=" * 50)
    print("Testing parallel processor...")
    print("=" * 50)
    
    try:
        from backend.models.parallel_processor import ParallelProcessor
        
        processor = ParallelProcessor()
        
        test_text = "Technology stocks surged today as Apple and Google announced new AI features."
        
        result = processor.process(
            text=test_text,
            models=['text', 'audio', 'image', 'video']
        )
        
        print(f"Success: {result.success}")
        print(f"Primary category: {result.primary_category}")
        print(f"Confidence: {result.confidence}")
        print(f"Total processing time: {result.total_processing_time:.3f}s")
        print(f"Partial failures: {result.partial_failures}")
        
        for model_type, model_result in result.model_results.items():
            print(f"  {model_type}: success={model_result.success}, category={model_result.primary_category}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Parallel processor failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("PARALLEL PROCESSING SYSTEM TEST")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Text Worker", test_text_worker),
        ("Audio Worker", test_audio_worker),
        ("Image Worker", test_image_worker),
        ("Video Worker", test_video_worker),
        ("Parallel Processor", test_parallel_processor),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"[FAIL] {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
