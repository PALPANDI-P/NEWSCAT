#!/usr/bin/env python
"""Test API endpoints"""
import requests
import json

BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/api/health")
    print("Health Check:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_classify(text, enhanced=True):
    """Test classification endpoint"""
    payload = {
        "text": text,
        "enhanced": enhanced
    }
    response = requests.post(f"{BASE_URL}/api/classify", json=payload)
    print(f"Classification (enhanced={enhanced}):")
    print(json.dumps(response.json(), indent=2))
    print()

def test_keywords(text):
    """Test keyword extraction endpoint"""
    payload = {
        "text": text,
        "top_n": 5
    }
    response = requests.post(f"{BASE_URL}/api/keywords", json=payload)
    print("Keywords:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_categories():
    """Test categories endpoint"""
    response = requests.get(f"{BASE_URL}/api/categories")
    print("Categories:")
    print(json.dumps(response.json(), indent=2))
    print()

if __name__ == "__main__":
    print("="*60)
    print("NEWSCAT Phase 2 API Test")
    print("="*60)
    print()
    
    # Test health
    test_health()
    
    # Test categories
    test_categories()
    
    # Test classification
    test_texts = [
        "Apple announced the release of iPhone 15 with revolutionary AI capabilities that will transform mobile computing and machine learning applications.",
        "Manchester United defeated Liverpool 3-1 in a thrilling Premier League match at Old Trafford.",
        "The Senate passed a landmark $500 billion climate change bill with bipartisan support."
    ]
    
    for text in test_texts:
        test_classify(text, enhanced=True)
    
    # Test keywords
    test_keywords(test_texts[0])
    
    print("="*60)
    print("All tests completed!")
    print("="*60)