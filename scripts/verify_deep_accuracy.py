import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from backend.models.simple_classifier import SimpleNewsClassifier
from backend.response_formatter import format_classification_result
from backend.utils import TextValidator

def test_large_text_validation():
    print("\n--- [TEST] LARGE TEXT VALIDATION (HTTP 400 FIX) ---")
    # Generate 60,000 character string
    large_text = "This is a news report about a very long event. " * 1500
    print(f"Testing text length: {len(large_text)}")
    
    is_valid, result = TextValidator.is_valid(large_text)
    if is_valid:
        print("SUCCESS: 60,000 char text is VALID (Fix working!)")
    else:
        print(f"FAILED: 60,000 char text REJECTED: {result}")
        assert is_valid

def test_cricket_naming():
    print("\n--- [TEST] CRICKET NAMING (CORE VS MAIN) ---")
    classifier = SimpleNewsClassifier()
    text = "The IPL final between Mumbai Indians and CSK reached a thrilling super over after a hat-trick by the bowler."
    result = classifier.classify(text)
    
    res = format_classification_result(
        category=result["category"],
        confidence=result["confidence"],
        input_type="text",
        category_display=result["category_display"]
    )
    
    core = res['data']['core_topic']
    main = res['data']['content_main_topic']
    
    print(f"Input: {text}")
    print(f"RESULT_CORE: {core}")
    print(f"RESULT_MAIN: {main}")
    
    assert core != main, "Core and Main topics should be distinct!"
    assert "Sport" in main, f"Main topic should be Sports related, got {main}"

def test_ai_accuracy():
    print("\n--- [TEST] AI ACCURACY (FUTURISTIC LOGIC) ---")
    classifier = SimpleNewsClassifier()
    # Complex AI news with RAG, fine-tuning, etc.
    text = "Improving LLM inference using RAG and specialized fine-tuning on a massive compute cluster."
    result = classifier.classify(text)
    
    res = format_classification_result(
        category=result["category"],
        confidence=result["confidence"],
        input_type="text",
        category_display=result["category_display"]
    )
    
    print(f"Input: {text}")
    print(f"Core: {res['data']['core_topic']}")
    print(f"Main: {res['data']['content_main_topic']}")
    print(f"Confidence: {res['data']['confidence']}%")
    
    assert "Intelligence" in res['data']['core_topic'] or "AI" in res['data']['core_topic']
    assert res['data']['confidence'] > 70, "AI confidence should be high for specialized keywords!"

if __name__ == "__main__":
    try:
        test_large_text_validation()
        test_cricket_naming()
        test_ai_accuracy()
        print("\n=== DEEP ACCURACY VERIFIED! ===")
    except Exception as e:
        print(f"\nDEEP VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
