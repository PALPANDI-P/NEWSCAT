"""
Advanced Text Processing Module
Handles all text preprocessing and feature extraction
"""

import re
import string
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
import logging

logger = logging.getLogger(__name__)

# Try to import NLTK, fallback to basic processing if not available
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    
    # Download NLTK data if needed
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    
    NLTK_AVAILABLE = True
    logger.info("NLTK loaded successfully")
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available, using basic processing")


class TextProcessor:
    """Advanced text processing with multiple techniques"""
    
    def __init__(self, use_advanced: bool = True, config: Dict = None):
        self.use_advanced = use_advanced and NLTK_AVAILABLE
        self.config = config or {}
        
        if self.use_advanced:
            self.stop_words = set(stopwords.words('english'))
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
            logger.info("Advanced text processor initialized")
        else:
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                              'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was',
                              'are', 'were', 'be', 'been', 'being'}
            logger.info("Basic text processor initialized")
    
    def clean_text(self, text: str, remove_numbers: bool = True) -> str:
        """
        Comprehensive text cleaning
        
        Args:
            text: Input text
            remove_numbers: Whether to remove numbers
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove numbers if requested
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str, remove_stopwords: bool = True,
                min_length: int = 3) -> List[str]:
        """
        Advanced tokenization
        
        Args:
            text: Input text
            remove_stopwords: Remove common stopwords
            min_length: Minimum token length
            
        Returns:
            List of tokens
        """
        # Clean text first
        cleaned = self.clean_text(text)
        
        if not cleaned:
            return []
        
        # Tokenize
        if self.use_advanced:
            try:
                tokens = word_tokenize(cleaned)
            except:
                tokens = cleaned.split()
        else:
            tokens = cleaned.split()
        
        # Filter tokens
        filtered = []
        for token in tokens:
            # Check length
            if len(token) < min_length:
                continue
            
            # Remove stopwords
            if remove_stopwords and token in self.stop_words:
                continue
            
            # Remove single characters
            if len(token) == 1 and token not in 'ai':
                continue
            
            filtered.append(token)
        
        return filtered
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Apply lemmatization to tokens"""
        if not self.use_advanced:
            return tokens
        
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def get_ngrams(self, tokens: List[str], n: int = 2) -> List[str]:
        """Generate n-grams from tokens"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i + n])
            ngrams.append(ngram)
        return ngrams
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """
        Extract comprehensive text features
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Sentence tokenization
        if self.use_advanced:
            sentences = sent_tokenize(text)
        else:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Word tokenization
        words = self.tokenize(text, remove_stopwords=False)
        content_words = self.tokenize(text, remove_stopwords=True)
        
        # Length features
        features['char_count'] = len(text)
        features['word_count'] = len(words)
        features['sentence_count'] = len(sentences)
        features['avg_word_length'] = float(np.mean([len(w) for w in words])) if words else 0
        features['avg_sentence_length'] = len(words) / len(sentences) if sentences else 0
        
        # Vocabulary features
        unique_words = set(content_words)
        features['unique_words'] = len(unique_words)
        features['lexical_diversity'] = len(unique_words) / len(content_words) if content_words else 0
        
        # Part of speech ratios (simplified)
        if content_words:
            # Detect nouns (words ending in specific patterns)
            nouns = sum(1 for w in content_words if w.endswith(('tion', 'ment', 'ity', 'ness')))
            features['noun_ratio'] = nouns / len(content_words)
            
            # Detect verbs
            verbs = sum(1 for w in content_words if w.endswith(('ed', 'ing', 'ate', 'ize')))
            features['verb_ratio'] = verbs / len(content_words)
            
            # Detect adjectives
            adj = sum(1 for w in content_words if w.endswith(('able', 'ible', 'ful', 'ous')))
            features['adjective_ratio'] = adj / len(content_words)
        
        # Text quality features
        features['capital_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        
        # Readability score
        features['readability'] = self._calculate_readability(text, sentences, words)
        
        return features
    
    def _calculate_readability(self, text: str, sentences: List[str], words: List[str]) -> float:
        """Calculate Flesch Reading Ease score"""
        if len(sentences) == 0 or len(words) == 0:
            return 0
        
        avg_sentence = len(words) / len(sentences)
        syllable_count = self._count_syllables(text)
        avg_syllables = syllable_count / len(words) if words else 0
        
        flesch = 206.835 - (1.015 * avg_sentence) - (84.6 * avg_syllables)
        return float(max(0, min(100, flesch)))
    
    def _count_syllables(self, text: str) -> int:
        """Simple syllable counter"""
        words = self.tokenize(text, remove_stopwords=False)
        count = 0
        
        for word in words:
            word = word.lower()
            syllables = 0
            vowels = 'aeiouy'
            
            if word and word[0] in vowels:
                syllables += 1
            
            for i in range(1, len(word)):
                if word[i] in vowels and word[i-1] not in vowels:
                    syllables += 1
            
            if word.endswith('e'):
                syllables -= 1
            
            count += max(1, syllables)
        
        return count
    
    def get_keywords(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords using TF-IDF approximation
        
        Args:
            text: Input text
            top_n: Number of keywords to return
            
        Returns:
            List of (keyword, score) tuples
        """
        words = self.tokenize(text, remove_stopwords=True)
        
        if not words:
            return []
        
        # Simple frequency-based scoring
        word_freq = Counter(words)
        max_freq = max(word_freq.values())
        
        return [(word, count / max_freq) for word, count in word_freq.most_common(top_n)]
    
    def preprocess_pipeline(self, text: str) -> str:
        """
        Complete preprocessing pipeline
        
        Args:
            text: Input text
            
        Returns:
            Processed text
        """
        # Clean text
        cleaned = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned, remove_stopwords=True)
        
        # Lemmatize
        if self.use_advanced:
            tokens = self.lemmatize_tokens(tokens)
        
        return ' '.join(tokens)