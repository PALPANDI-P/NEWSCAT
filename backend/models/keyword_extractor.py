"""
KeywordExtractor - Simple keyword extraction stub
"""

class KeywordExtractor:
    def __init__(self):
        pass

    def extract(self, text, top_n=5):
        # Stub: Returns first N unique words
        words = list(dict.fromkeys(text.split()))
        return words[:top_n]

    def get_info(self):
        return {
            "name": "KeywordExtractor",
            "description": "Extracts keywords from text (stub implementation)."
        }
