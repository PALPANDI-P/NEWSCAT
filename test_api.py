"""
NEWSCAT API Tests
Basic API endpoint testing for the news classification system
"""

import pytest
import requests
import json
from unittest.mock import Mock, patch

# Test configuration
BASE_URL = "http://127.0.0.1:5000"
TEST_TIMEOUT = 10

class TestAPIEndpoints:
    """Test suite for API endpoints"""

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = requests.get(f"{BASE_URL}/api/health", timeout=TEST_TIMEOUT)
        assert response.status_code == 200

        data = response.json()
        assert data['status'] == 'success'
        assert 'classifications_available' in data
        assert 'version' in data

    def test_categories_endpoint(self):
        """Test categories listing endpoint"""
        response = requests.get(f"{BASE_URL}/api/categories", timeout=TEST_TIMEOUT)
        assert response.status_code == 200

        data = response.json()
        assert data['status'] == 'success'
        assert 'categories' in data
        assert isinstance(data['categories'], list)
        assert len(data['categories']) > 0

    def test_classify_text_endpoint(self):
        """Test text classification endpoint"""
        test_text = "Apple reported record quarterly earnings today, beating analyst expectations."

        response = requests.post(
            f"{BASE_URL}/api/classify",
            json={"text": test_text},
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 200

        data = response.json()
        assert data['status'] == 'success'
        assert 'category' in data
        assert 'confidence' in data
        assert 'model_name' in data

    def test_classify_text_validation(self):
        """Test text validation"""
        # Test empty text
        response = requests.post(
            f"{BASE_URL}/api/classify",
            json={"text": ""},
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 400

        # Test too short text
        response = requests.post(
            f"{BASE_URL}/api/classify",
            json={"text": "Hi"},
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 400

    def test_model_info_endpoint(self):
        """Test model info endpoint"""
        response = requests.get(f"{BASE_URL}/api/model/info", timeout=TEST_TIMEOUT)
        assert response.status_code == 200

        data = response.json()
        assert data['status'] == 'success'
        assert 'model_name' in data
        assert 'model_version' in data

    @pytest.mark.parametrize("endpoint", [
        "/api/health",
        "/api/categories",
        "/api/model/info"
    ])
    def test_cors_headers(self, endpoint):
        """Test CORS headers are present"""
        response = requests.options(f"{BASE_URL}{endpoint}", timeout=TEST_TIMEOUT)
        assert 'Access-Control-Allow-Origin' in response.headers
        assert 'Access-Control-Allow-Methods' in response.headers
        assert 'Access-Control-Allow-Headers' in response.headers

class TestErrorHandling:
    """Test error handling scenarios"""

    def test_invalid_json(self):
        """Test handling of invalid JSON"""
        response = requests.post(
            f"{BASE_URL}/api/classify",
            data="invalid json",
            headers={'Content-Type': 'application/json'},
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 400

    def test_missing_text_field(self):
        """Test handling of missing text field"""
        response = requests.post(
            f"{BASE_URL}/api/classify",
            json={"content": "some text"},  # wrong field name
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 400

    def test_nonexistent_endpoint(self):
        """Test 404 for nonexistent endpoints"""
        response = requests.get(f"{BASE_URL}/api/nonexistent", timeout=TEST_TIMEOUT)
        assert response.status_code == 404

if __name__ == "__main__":
    # Run basic smoke tests
    print("Running NEWSCAT API smoke tests...")

    try:
        # Test health endpoint
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            print("[OK] Health check passed")
        else:
            print(f"[FAIL] Health check failed: {response.status_code}")

        # Test categories endpoint
        response = requests.get(f"{BASE_URL}/api/categories", timeout=5)
        if response.status_code == 200:
            print("[OK] Categories endpoint passed")
        else:
            print(f"[FAIL] Categories endpoint failed: {response.status_code}")

        # Test classify endpoint
        test_text = "Apple reported record quarterly earnings today, beating analyst expectations."
        response = requests.post(
            f"{BASE_URL}/api/classify",
            json={"text": test_text},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            print(f"[OK] Classification endpoint passed - Category: {data.get('category')}, Confidence: {data.get('confidence')}")
        else:
            print(f"[FAIL] Classification endpoint failed: {response.status_code}")

        print("Basic smoke tests completed. Run with pytest for full test suite.")

    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot connect to server. Make sure the application is running.")
    except Exception as e:
        print(f"[ERROR] Test error: {e}")