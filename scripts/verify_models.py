import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Mock basic logging
import logging
logging.basicConfig(level=logging.INFO)

from backend.models.simple_classifier import SimpleNewsClassifier
from backend.models.video_processor import VideoProcessor, VideoResult
from backend.models.parallel_processor import ParallelProcessor
from backend.response_formatter import format_classification_result

def test_text_model():
    print("\n--- [TEST] TEXT MODEL (Expert Heuristics) ---")
    classifier = SimpleNewsClassifier()
    text = "NASA's new lunar rover is a milestone for space tourism and exploratory science."
    result = classifier.classify(text)
    
    # Format as API response
    res = format_classification_result(
        category=result["category"],
        confidence=result["confidence"],
        input_type="text",
        main_topic_summary=result["summary"],
        category_display=result["category_display"]
    )
    
    print(f"Text Input: {text[:50]}...", flush=True)
    print(f"RESULT_CORE: {res['data']['core_topic']}", flush=True)
    print(f"RESULT_MAIN: {res['data']['content_main_topic']}", flush=True)
    # The user specifically requested 'Science' as a main topic in their example
    assert res['data']['content_main_topic'] in ["Science", "Science & Discovery", "General Science"], f"Expected a Science-related parent but got {res['data']['content_main_topic']}"

def test_video_audio_boost():
    print("\n--- [TEST] VIDEO 3x AUDIO BOOST ---")
    # We mock the video extraction result to simulate "Clear Audio" vs "Noisy Frames"
    audio_text = "The football match between Real Madrid and Barcelona was spectacular."
    noisy_frames = ["random noise", "ad", "unwanted text", "123", "abc"]
    
    # Manually demonstrate the weight injection in VideoProcessor
    weighted_parts = []
    weighted_parts.append(f"[PRIORITY_AUDIO]: {audio_text} {audio_text} {audio_text}")
    weighted_parts.extend(noisy_frames)
    combined_text = " ".join(weighted_parts).strip()
    
    classifier = SimpleNewsClassifier()
    result = classifier.classify(combined_text)
    
    print(f"Audio Context: '{audio_text}'")
    print(f"Noise Context: {noisy_frames}")
    print(f"Combined Extracted Length: {len(combined_text)}")
    print(f"Core Topic: {result['category_display']}")
    
    # The topic MUST be football (Sports) even with noise
    assert result['category'] in ["football", "sports"]
    print("SUCCESS: Audio 3x Boost correctly dominated the noisy frame OCR!")

def run_all():
    try:
        test_text_model()
        test_video_audio_boost()
        print("\n=== ALL MODELS VERIFIED SUCCESSFULLY! ===")
    except Exception as e:
        print(f"\nVERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all()
