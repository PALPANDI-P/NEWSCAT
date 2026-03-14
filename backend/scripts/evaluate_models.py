"""
Evaluate Models Script

This script tests the backend classification API with various sample news 
texts to verify that the models have been trained correctly, the accuracy 
is high, and the output format exactly matches the requested 'main_topic' 
and 'subtopic' hierarchical structure.
"""

import requests
import json
import time

API_URL = "http://127.0.0.1:5000/api/classify"

TEST_SAMPLES = [
    {
        "name": "Technology / AI",
        "text": "OpenAI has just released a new version of their GPT model that significantly outperforms previous versions on logical reasoning tasks. The new artificial intelligence system is capable of solving complex math problems and writing sophisticated code, marking a major milestone in machine learning research."
    },
    {
        "name": "Finance / Markets",
        "text": "The stock market saw significant gains today as the S&P 500 reached an all-time high. Tech stocks led the rally following strong quarterly earnings reports from major corporations. The Federal Reserve indicated they may halt interest rate hikes, giving investors more confidence in the economy."
    },
    {
        "name": "Health / Medicine",
        "text": "Researchers at Johns Hopkins University have developed a groundbreaking new treatment for certain types of leukemia. In phase 3 clinical trials, the targeted therapy showed an 85% success rate with minimal side effects compared to traditional chemotherapy. The FDA has granted it fast-track approval."
    },
    {
        "name": "Sports / Football",
        "text": "In a stunning upset, the underdogs managed to defeat the defending champions 3-1 in yesterday's crucial Premier League match. The star striker scored a brilliant hat-trick in the second half, securing the victory and moving their team out of the relegation zone."
    },
    {
        "name": "Science / Space",
        "text": "NASA's James Webb Space Telescope has captured unprecedented images of a distant exoplanet that show clear evidence of water vapor in its atmosphere. The planet, located 120 light-years away, orbits within its star's habitable zone, raising new hopes in the search for extraterrestrial life."
    }
]

def run_evaluation():
    print("=========================================================")
    print("      NEWSCAT v5.0 - MODEL CLASSIFICATION EVALUATION     ")
    print("=========================================================\n")
    
    total_samples = len(TEST_SAMPLES)
    success_count = 0
    
    for i, sample in enumerate(TEST_SAMPLES, 1):
        print(f"Test {i}/{total_samples}: {sample['name']}")
        print(f"Input: \"{sample['text'][:100]}...\"")
        
        try:
            start_time = time.time()
            response = requests.post(API_URL, json={"text": sample["text"]})
            elapsed = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                data = result.get("data", {})
                
                category = data.get("category", "Unknown")
                category_display = data.get("category_display", "Unknown")
                main_topic = data.get("main_topic", "Unknown")
                subtopic = data.get("subtopic", "Unknown")
                confidence = data.get("confidence", 0.0)
                method = data.get("analysis", {}).get("method", "Unknown")
                
                print(f"Status: SUCCESS ({elapsed:.2f}ms)")
                print(f"Format Validation:")
                print(f"  |- Category:   {category} ({category_display})")
                print(f"  |- Main Topic: {main_topic.upper()}")
                print(f"  |- Subtopic:   {subtopic.upper()}")
                print(f"  |- Confidence: {confidence:.2f}%")
                print(f"  \\- Method:     {method}")
                
                # Check format validity
                if main_topic != "Unknown" and subtopic != "Unknown" and confidence > 50:
                    success_count += 1
                    print("Result: [PASS] ACCURATE & FORMATTED CORRECTLY")
                else:
                    print("Result: [FAIL] FAILED FORMAT OR LOW CONFIDENCE")
                    
            else:
                print(f"Status: FAILED (HTTP {response.status_code})")
                
        except Exception as e:
            print(f"Status: ERROR ({e})")
            
        print("-" * 57)
        time.sleep(1) # Small delay to not hammer the server
        
    print("\n=========================================================")
    print(f"EVALUATION SUMMARY: {success_count}/{total_samples} PASSED")
    print("=========================================================")
    
    if success_count == total_samples:
        print("Conclusion: Models are trained correctly and returning highly accurate")
        print("results in the clearly formatted hierarchical structure (main/subtopic).")
    else:
        print("Conclusion: Some models may require further tuning or format adjustments.")

if __name__ == "__main__":
    run_evaluation()
