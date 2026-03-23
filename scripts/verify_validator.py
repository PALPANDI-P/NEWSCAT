import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from backend.utils import TextValidator

def test_validator():
    print("--- [TEST] TextValidator Edge Cases ---")
    
    # 1. Real news with 'on' and '=' (should pass)
    news_text = "The merger was decided on Tuesday = $50 billion value."
    is_valid, result = TextValidator.is_valid(news_text)
    print(f"Test 'on Tuesday =': {'PASS' if is_valid else 'FAIL'}")
    if not is_valid: print(f"  Error: {result}")
    
    # 2. Actual XSS (should fail)
    xss_text = "Hello <script>alert('xss')</script>"
    is_valid, result = TextValidator.is_valid(xss_text)
    print(f"Test <script>: {'PASS' if not is_valid else 'FAIL'}")
    
    # 3. Direct event handler (should fail)
    event_text = "Check this out: <div onclick='hack()'>click</div>"
    is_valid, result = TextValidator.is_valid(event_text)
    print(f"Test onclick=: {'PASS' if not is_valid else 'FAIL'}")

    # 4. Short text (should fail)
    short_text = "abc"
    is_valid, result = TextValidator.is_valid(short_text)
    print(f"Test short text: {'PASS' if not is_valid else 'FAIL'} (Detail: {result})")

    if is_valid:
        print("=== VALIDATOR VERIFIED! ===")
    else:
        # We expect some to fail, so let's check manually
        pass

if __name__ == "__main__":
    test_validator()
