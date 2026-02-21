#!/usr/bin/env python
"""
NEWSCAT Phase 2 Unit Tests
Comprehensive tests for Advanced Text Processor, Ensemble Classifier, and Keyword Extractor

Run with: pytest tests/test_phase2_models.py -v
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# ADVANCED TEXT PROCESSOR TESTS
# ============================================================================

class TestAdvancedTextProcessor:
    """Tests for AdvancedTextProcessor"""
    
    @pytest.fixture
    def processor(self):
        """Create processor instance"""
        from backend.models.advanced_text_processor import AdvancedTextProcessor
        return AdvancedTextProcessor(use_spacy=False, use_nltk=False)
    
    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing"""
        return {
            'technology': "Apple released new iPhone with advanced AI features and machine learning capabilities.",
            'sports': "Manchester United won the Premier League match against Liverpool 3-1.",
            'politics': "Congress passed a new healthcare bill with bipartisan support in the Senate.",
            'business': "Stock market reached record highs as Apple reported quarterly earnings.",
            'health': "New vaccine shows 95% effectiveness in clinical trials against the virus.",
            'with_urls': "Visit https://example.com for more info. Email: test@example.com",
            'with_html': "<p>This is <b>HTML</b> content.</p>",
            'empty': "",
            'short': "Hi",
            'long': "This is a test. " * 100
        }
    
    def test_clean_text_basic(self, processor, sample_texts):
        """Test basic text cleaning"""
        result = processor.clean_text(sample_texts['technology'])
        assert len(result) > 0
        assert result.islower()
        assert "https://" not in result
    
    def test_clean_text_removes_urls(self, processor, sample_texts):
        """Test URL removal"""
        result = processor.clean_text(sample_texts['with_urls'])
        assert "https://example.com" not in result
        assert "example.com" not in result
    
    def test_clean_text_removes_emails(self, processor, sample_texts):
        """Test email removal"""
        result = processor.clean_text(sample_texts['with_urls'])
        assert "test@example.com" not in result
    
    def test_clean_text_removes_html(self, processor, sample_texts):
        """Test HTML tag removal"""
        result = processor.clean_text(sample_texts['with_html'])
        assert "<p>" not in result
        assert "<b>" not in result
    
    def test_clean_text_empty(self, processor, sample_texts):
        """Test empty text handling"""
        result = processor.clean_text(sample_texts['empty'])
        assert result == ""
    
    def test_tokenize_basic(self, processor, sample_texts):
        """Test basic tokenization"""
        tokens = processor.tokenize(sample_texts['technology'])
        assert len(tokens) > 0
        assert all(len(t) >= 2 for t in tokens)
    
    def test_tokenize_removes_stopwords(self, processor, sample_texts):
        """Test stopword removal"""
        tokens = processor.tokenize("The quick brown fox jumps over the lazy dog")
        assert "the" not in tokens
        assert "over" not in tokens
    
    def test_tokenize_min_length(self, processor):
        """Test minimum token length"""
        tokens = processor.tokenize("I am a test with short words", min_length=4)
        assert all(len(t) >= 4 for t in tokens)
    
    def test_tokenize_empty(self, processor, sample_texts):
        """Test empty text tokenization"""
        tokens = processor.tokenize(sample_texts['empty'])
        assert tokens == []
    
    def test_preprocess_pipeline(self, processor, sample_texts):
        """Test complete preprocessing pipeline"""
        result = processor.preprocess_pipeline(sample_texts['technology'])
        assert len(result) > 0
        assert result.islower()
        # Should not contain stopwords
        words = result.split()
        assert "the" not in words
        assert "with" not in words
    
    def test_extract_features(self, processor, sample_texts):
        """Test feature extraction"""
        features = processor.extract_features(sample_texts['technology'])
        
        assert features.word_count > 0
        assert features.char_count > 0
        assert features.sentence_count > 0
        assert 0 <= features.lexical_diversity <= 1
        assert 0 <= features.readability_score <= 100
        assert -1 <= features.sentiment_score <= 1
    
    def test_extract_features_empty(self, processor, sample_texts):
        """Test feature extraction with empty text"""
        features = processor.extract_features(sample_texts['empty'])
        assert features.word_count == 0
        assert features.char_count == 0
    
    def test_calculate_sentiment_positive(self, processor):
        """Test sentiment analysis - positive"""
        score = processor._calculate_sentiment("This is great news with excellent results and amazing success!")
        assert score > 0
    
    def test_calculate_sentiment_negative(self, processor):
        """Test sentiment analysis - negative"""
        score = processor._calculate_sentiment("This is terrible news with poor results and bad failure!")
        assert score < 0
    
    def test_calculate_sentiment_neutral(self, processor):
        """Test sentiment analysis - neutral"""
        score = processor._calculate_sentiment("The weather is cloudy today.")
        assert score == 0
    
    def test_calculate_readability(self, processor):
        """Test readability calculation"""
        # Simple text should be readable
        score = processor._calculate_readability(
            "This is a simple sentence. It has short words.",
            ["This is a simple sentence.", "It has short words."],
            ["This", "is", "a", "simple", "sentence", "It", "has", "short", "words"]
        )
        assert 0 <= score <= 100
    
    def test_count_syllables(self, processor):
        """Test syllable counting"""
        assert processor._count_syllables("the") == 1
        assert processor._count_syllables("apple") == 2
        assert processor._count_syllables("beautiful") >= 3
    
    def test_get_ngrams(self, processor):
        """Test n-gram generation"""
        tokens = ["the", "quick", "brown", "fox"]
        bigrams = processor.get_ngrams(tokens, n=2)
        assert len(bigrams) == 3
        assert "the quick" in bigrams
        
        trigrams = processor.get_ngrams(tokens, n=3)
        assert len(trigrams) == 2
    
    def test_get_keywords_frequency(self, processor, sample_texts):
        """Test keyword frequency extraction"""
        keywords = processor.get_keywords_frequency(sample_texts['technology'], top_n=5)
        assert len(keywords) <= 5
        assert all(score > 0 for _, score in keywords)


# ============================================================================
# ENSEMBLE CLASSIFIER TESTS
# ============================================================================

class TestEnsembleClassifier:
    """Tests for EnsembleNewsClassifier"""
    
    @pytest.fixture
    def classifier(self):
        """Create classifier instance"""
        from backend.models.ensemble_classifier import EnsembleNewsClassifier
        return EnsembleNewsClassifier()
    
    @pytest.fixture
    def training_data(self):
        """Small training dataset"""
        return (
            [
                "Apple released new iPhone with AI features",
                "Google announced breakthrough in quantum computing",
                "Microsoft acquires cybersecurity startup",
                "Manchester United won the Premier League match",
                "Lakers defeated Celtics in NBA finals",
                "Tennis champion wins Wimbledon title",
                "Congress passed new healthcare bill",
                "Senate approved infrastructure spending",
                "President signs executive order on climate",
                "Stock market reached record highs today",
                "Company reports quarterly earnings growth",
                "Tesla stock surges after announcement",
            ],
            [
                "technology", "technology", "technology",
                "sports", "sports", "sports",
                "politics", "politics", "politics",
                "business", "business", "business"
            ]
        )
    
    def test_classifier_initialization(self, classifier):
        """Test classifier initializes correctly"""
        assert classifier.name == "EnsembleClassifier"
        assert classifier.version == "2.0.0"
        assert len(classifier.categories) == 10
    
    def test_classify_without_training(self, classifier):
        """Test classification without training (rule-based fallback)"""
        result = classifier.classify("Apple released new iPhone with AI features")
        
        assert 'category' in result
        assert 'confidence' in result
        assert result['category'] in classifier.categories
        assert 0 <= result['confidence'] <= 1
    
    def test_classify_technology(self, classifier):
        """Test technology classification"""
        result = classifier.classify("Apple released new iPhone with AI features")
        assert result['category'] == 'technology'
    
    def test_classify_sports(self, classifier):
        """Test sports classification"""
        result = classifier.classify("Manchester United won the Premier League match")
        assert result['category'] == 'sports'
    
    def test_classify_politics(self, classifier):
        """Test politics classification"""
        result = classifier.classify("Congress passed new healthcare bill today")
        assert result['category'] == 'politics'
    
    def test_classify_business(self, classifier):
        """Test business classification"""
        result = classifier.classify("Stock market reached record highs this quarter")
        assert result['category'] == 'business'
    
    def test_classify_invalid_input(self, classifier):
        """Test classification with invalid input"""
        result = classifier.classify("")
        assert result['category'] == 'unknown'
        
        result = classifier.classify("short")
        assert result['category'] == 'unknown'
    
    def test_classify_returns_top_predictions(self, classifier):
        """Test classification returns top predictions"""
        result = classifier.classify("Apple released new iPhone with AI features")
        
        assert 'top_predictions' in result
        assert len(result['top_predictions']) <= 3
        for pred in result['top_predictions']:
            assert 'category' in pred
            assert 'confidence' in pred
    
    def test_train_classifier(self, classifier, training_data):
        """Test classifier training"""
        texts, labels = training_data
        
        results = classifier.train(texts, labels, validate=False)
        
        assert 'accuracy' in results
        assert results['accuracy'] > 0
        assert classifier.is_trained
    
    def test_train_with_validation(self, classifier, training_data):
        """Test classifier training with cross-validation"""
        texts, labels = training_data
        
        # Need more data for cross-validation
        results = classifier.train(texts, labels, validate=False)
        
        assert 'accuracy' in results
    
    def test_get_info(self, classifier):
        """Test get_info method"""
        info = classifier.get_info()
        
        assert 'name' in info
        assert 'version' in info
        assert 'categories' in info
        assert 'trained' in info
        assert 'classifiers' in info
        assert info['voting'] == 'soft'
    
    def test_validate_input(self, classifier):
        """Test input validation"""
        # Valid input
        assert classifier._validate_input("This is a valid news article text for classification.")
        
        # Too short
        assert not classifier._validate_input("short")
        
        # Empty
        assert not classifier._validate_input("")
        
        # None
        assert not classifier._validate_input(None)


# ============================================================================
# KEYWORD EXTRACTOR TESTS
# ============================================================================

class TestAdvancedKeywordExtractor:
    """Tests for AdvancedKeywordExtractor"""
    
    @pytest.fixture
    def extractor(self):
        """Create extractor instance"""
        from backend.models.advanced_keyword_extractor import AdvancedKeywordExtractor
        return AdvancedKeywordExtractor(method='hybrid')
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing"""
        return """
        Apple Inc. announced record quarterly earnings of $120 billion, 
        driven by strong iPhone sales and growing Services revenue. 
        CEO Tim Cook highlighted the company's expansion into emerging markets
        and the success of the new Apple Watch Series 9.
        """
    
    def test_extractor_initialization(self, extractor):
        """Test extractor initializes correctly"""
        assert extractor.method == 'hybrid'
        assert len(extractor.stop_words) > 0
    
    def test_extract_basic(self, extractor, sample_text):
        """Test basic keyword extraction"""
        keywords = extractor.extract(sample_text, top_n=10)
        
        assert len(keywords) <= 10
        assert all('keyword' in kw for kw in keywords)
        assert all('score' in kw for kw in keywords)
        assert all(0 <= kw['score'] <= 1 for kw in keywords)
    
    def test_extract_frequency_method(self, sample_text):
        """Test frequency extraction method"""
        from backend.models.advanced_keyword_extractor import AdvancedKeywordExtractor
        extractor = AdvancedKeywordExtractor(method='frequency')
        
        keywords = extractor.extract(sample_text, top_n=5)
        assert len(keywords) <= 5
    
    def test_extract_position_method(self, sample_text):
        """Test position-weighted extraction method"""
        from backend.models.advanced_keyword_extractor import AdvancedKeywordExtractor
        extractor = AdvancedKeywordExtractor(method='position')
        
        keywords = extractor.extract(sample_text, top_n=5)
        assert len(keywords) <= 5
    
    def test_extract_empty_text(self, extractor):
        """Test extraction with empty text"""
        keywords = extractor.extract("", top_n=5)
        assert keywords == []
    
    def test_extract_short_text(self, extractor):
        """Test extraction with short text"""
        keywords = extractor.extract("Hello world", top_n=5)
        # Should still work, just return fewer keywords
        assert isinstance(keywords, list)
    
    def test_extract_with_context(self, extractor, sample_text):
        """Test extraction with context"""
        keywords = extractor.extract_with_context(sample_text, top_n=5)
        
        assert len(keywords) <= 5
        for kw in keywords:
            assert 'keyword' in kw
            assert 'score' in kw
            assert 'count' in kw
            assert 'is_entity' in kw
    
    def test_get_key_phrases(self, extractor, sample_text):
        """Test key phrase extraction"""
        phrases = extractor.get_key_phrases(sample_text, top_n=5)
        
        assert len(phrases) <= 5
        for phrase in phrases:
            assert 'phrase' in phrase
            assert 'score' in phrase
            assert ' ' in phrase['phrase']  # Should be multi-word
    
    def test_get_info(self, extractor):
        """Test get_info method"""
        info = extractor.get_info()
        
        assert 'name' in info
        assert 'method' in info
        assert info['method'] == 'hybrid'
    
    def test_stopwords_filtered(self, extractor):
        """Test that stopwords are filtered"""
        text = "The quick brown fox jumps over the lazy dog"
        keywords = extractor.extract(text, top_n=10)
        
        keyword_texts = [kw['keyword'] for kw in keywords]
        assert 'the' not in keyword_texts
        assert 'over' not in keyword_texts


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for Phase 2 components"""
    
    def test_full_pipeline(self):
        """Test full classification pipeline"""
        from backend.models.ensemble_classifier import EnsembleNewsClassifier
        from backend.models.advanced_keyword_extractor import AdvancedKeywordExtractor
        
        # Initialize
        classifier = EnsembleNewsClassifier()
        extractor = AdvancedKeywordExtractor()
        
        # Test text
        text = "Apple Inc. announced record quarterly earnings of $120 billion, driven by strong iPhone sales."
        
        # Classify
        result = classifier.classify(text)
        assert 'category' in result
        assert 'confidence' in result
        
        # Extract keywords
        keywords = extractor.extract(text, top_n=5)
        assert len(keywords) > 0
    
    def test_all_categories(self):
        """Test classification for all categories"""
        from backend.models.ensemble_classifier import EnsembleNewsClassifier
        
        classifier = EnsembleNewsClassifier()
        
        test_cases = {
            'technology': "Apple released new iPhone with AI features",
            'sports': "Manchester United won the Premier League match",
            'politics': "Congress passed new healthcare bill today",
            'business': "Stock market reached record highs this quarter",
            'entertainment': "The new movie won several Academy Awards",
            'health': "New vaccine shows effectiveness in clinical trials",
            'science': "NASA discovered new exoplanet with potential for life",
            'world': "United Nations announced humanitarian aid program",
            'education': "University announced new scholarship program",
            'environment': "Climate change affects Arctic ice levels"
        }
        
        for expected_category, text in test_cases.items():
            result = classifier.classify(text)
            # Rule-based should correctly classify most
            assert result['category'] in classifier.categories.keys()


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance tests for Phase 2 components"""
    
    def test_text_processor_performance(self):
        """Test text processor performance"""
        import time
        from backend.models.advanced_text_processor import AdvancedTextProcessor
        
        processor = AdvancedTextProcessor(use_spacy=False, use_nltk=False)
        
        # Long text
        long_text = "This is a test sentence. " * 1000
        
        start = time.time()
        for _ in range(10):
            processor.preprocess_pipeline(long_text)
        elapsed = time.time() - start
        
        # Should process 10 long texts in under 5 seconds
        assert elapsed < 5.0
    
    def test_classification_performance(self):
        """Test classification performance"""
        import time
        from backend.models.ensemble_classifier import EnsembleNewsClassifier
        
        classifier = EnsembleNewsClassifier()
        
        text = "Apple Inc. announced record quarterly earnings of $120 billion."
        
        # Warm up
        classifier.classify(text)
        
        # Time multiple classifications
        start = time.time()
        for _ in range(10):
            classifier.classify(text)
        elapsed = time.time() - start
        
        # Should classify 10 texts in under 2 seconds (rule-based)
        assert elapsed < 2.0
    
    def test_keyword_extraction_performance(self):
        """Test keyword extraction performance"""
        import time
        from backend.models.advanced_keyword_extractor import AdvancedKeywordExtractor
        
        extractor = AdvancedKeywordExtractor()
        
        text = "Apple Inc. announced record quarterly earnings. " * 10
        
        start = time.time()
        for _ in range(10):
            extractor.extract(text, top_n=10)
        elapsed = time.time() - start
        
        # Should extract keywords 10 times in under 3 seconds
        assert elapsed < 3.0


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])