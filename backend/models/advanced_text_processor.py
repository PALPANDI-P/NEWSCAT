"""
Advanced Text Processing Module
Enhanced with spaCy NLP pipeline for production-grade text processing

Performance: O(n) for most operations
"""

import re
import string
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
from dataclasses import dataclass
import logging

# Try imports with graceful fallback
try:
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TextFeatures:
    """Data class for extracted text features"""
    char_count: int
    word_count: int
    sentence_count: int
    avg_word_length: float
    avg_sentence_length: float
    unique_words: int
    lexical_diversity: float
    noun_count: int
    verb_count: int
    adjective_count: int
    entity_count: int
    sentiment_score: float
    readability_score: float
    exclamation_ratio: float
    question_ratio: float
    capital_ratio: float
    digit_ratio: float


class AdvancedTextProcessor:
    """
    Production-grade text processor with multiple NLP backends
    
    Features:
    - spaCy for tokenization, NER, POS tagging
    - NLTK fallback for lemmatization
    - Comprehensive text cleaning
    - Feature extraction for ML
    - Readability scoring
    - Sentiment analysis
    
    Usage:
        processor = AdvancedTextProcessor()
        processed = processor.preprocess_pipeline(text)
        features = processor.extract_features(text)
    """
    
    def __init__(self, use_spacy: bool = True, use_nltk: bool = True):
        """
        Initialize text processor
        
        Args:
            use_spacy: Try to use spaCy for NLP tasks
            use_nltk: Try to use NLTK as fallback
        """
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self.use_nltk = use_nltk and NLTK_AVAILABLE
        
        # Initialize spaCy
        if self.use_spacy:
            try:
                self.nlp = spacy.load('en_core_web_sm')
                self.stop_words = STOP_WORDS
                logger.info("spaCy model loaded successfully")
            except OSError:
                logger.warning("spaCy model not found, using basic processing")
                self.nlp = None
                self.stop_words = self._get_basic_stopwords()
                self.use_spacy = False
        else:
            self.nlp = None
            self.stop_words = self._get_basic_stopwords()
        
        # Initialize NLTK lemmatizer
        if self.use_nltk:
            try:
                self.lemmatizer = WordNetLemmatizer()
                nltk.download('punkt', quiet=True)
                nltk.download('wordnet', quiet=True)
                logger.info("NLTK initialized successfully")
            except:
                self.lemmatizer = None
                self.use_nltk = False
        else:
            self.lemmatizer = None
        
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.email_pattern = re.compile(r'\S+@\S+')
        self.html_pattern = re.compile(r'<[^>]+>')
        self.number_pattern = re.compile(r'\d+')
        self.whitespace_pattern = re.compile(r'\s+')
        
        logger.info(f"TextProcessor initialized (spacy={self.use_spacy}, nltk={self.use_nltk})")
    
    def _get_basic_stopwords(self) -> set:
        """Basic stopwords when spaCy not available"""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how',
            'not', 'so', 'if', 'then', 'than', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'under', 'again',
            'further', 'once', 'here', 'there', 'all', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'only', 'own', 'same',
            'too', 'very', 'just', 'also', 'now'
        }
    
    def clean_text(self, text: str, remove_numbers: bool = True) -> str:
        """
        Comprehensive text cleaning
        
        Operations:
        - Lowercase conversion
        - URL removal
        - Email removal
        - HTML tag removal
        - Number removal (optional)
        - Special character removal
        - Whitespace normalization
        
        Performance: O(n) where n is text length
        
        Args:
            text: Input text
            remove_numbers: Whether to remove numbers
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = self.url_pattern.sub('', text)
        
        # Remove emails
        text = self.email_pattern.sub('', text)
        
        # Remove HTML tags
        text = self.html_pattern.sub('', text)
        
        # Remove numbers if requested
        if remove_numbers:
            text = self.number_pattern.sub('', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s.!?]', ' ', text)
        
        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text).strip()
        
        return text
    
    def tokenize(self, text: str, remove_stopwords: bool = True, 
                 min_length: int = 2) -> List[str]:
        """
        Advanced tokenization with multiple backend support
        
        Performance: O(n) with spaCy, O(n log n) with NLTK
        
        Args:
            text: Input text
            remove_stopwords: Remove common stopwords
            min_length: Minimum token length
            
        Returns:
            List of tokens
        """
        cleaned = self.clean_text(text)
        
        if not cleaned:
            return []
        
        # Use spaCy for tokenization (faster and more accurate)
        if self.use_spacy and self.nlp:
            doc = self.nlp(cleaned)
            tokens = [
                token.text for token in doc
                if len(token.text) >= min_length
                and (not remove_stopwords or token.text.lower() not in self.stop_words)
                and not token.is_punct
                and not token.is_space
            ]
        # Fallback to NLTK
        elif self.use_nltk:
            try:
                tokens = word_tokenize(cleaned)
            except:
                tokens = cleaned.split()
            tokens = [
                t for t in tokens
                if len(t) >= min_length
                and (not remove_stopwords or t.lower() not in self.stop_words)
            ]
        # Basic tokenization
        else:
            tokens = cleaned.split()
            tokens = [
                t for t in tokens
                if len(t) >= min_length
                and (not remove_stopwords or t.lower() not in self.stop_words)
            ]
        
        return tokens
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatization with fallback
        
        Lemmatization converts words to their base form:
        - running -> run
        - better -> good
        - children -> child
        
        Performance: O(n) where n is number of tokens
        
        Args:
            tokens: List of tokens
            
        Returns:
            Lemmatized tokens
        """
        if not tokens:
            return []
        
        if self.use_spacy and self.nlp:
            # Process tokens through spaCy
            doc = self.nlp(' '.join(tokens))
            return [token.lemma_ for token in doc]
        
        if self.use_nltk and self.lemmatizer:
            return [self.lemmatizer.lemmatize(t) for t in tokens]
        
        # No lemmatization available
        return tokens
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Named Entity Recognition
        
        Extracts named entities like:
        - PERSON: Names of people
        - ORG: Organizations
        - GPE: Countries, cities
        - DATE: Dates and times
        - MONEY: Monetary values
        
        Performance: O(n) with spaCy
        
        Args:
            text: Input text
            
        Returns:
            List of entities with type and text
        """
        if not self.use_spacy or not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'description': spacy.explain(ent.label_) if ent.label_ else None
            })
        
        return entities
    
    def get_pos_tags(self, text: str) -> Dict[str, int]:
        """
        Part-of-speech tagging and counting
        
        Common POS tags:
        - NOUN: Nouns
        - VERB: Verbs
        - ADJ: Adjectives
        - ADV: Adverbs
        - PROPN: Proper nouns
        
        Performance: O(n) with spaCy
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of POS tag counts
        """
        if not self.use_spacy or not self.nlp:
            return {}
        
        doc = self.nlp(text)
        pos_counts = Counter()
        
        for token in doc:
            pos_counts[token.pos_] += 1
        
        return dict(pos_counts)
    
    def extract_features(self, text: str) -> TextFeatures:
        """
        Comprehensive feature extraction for ML models
        
        Extracts 17 features including:
        - Basic counts (chars, words, sentences)
        - Word statistics (avg length, unique words)
        - POS counts (nouns, verbs, adjectives)
        - Entity count
        - Sentiment score
        - Readability score
        - Punctuation ratios
        
        Performance: O(n) overall
        
        Args:
            text: Input text
            
        Returns:
            TextFeatures dataclass with all extracted features
        """
        # Basic counts
        char_count = len(text)
        words = text.split()
        word_count = len(words)
        
        # Sentence count
        if self.use_nltk:
            try:
                sentences = sent_tokenize(text)
            except:
                sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        else:
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        sentence_count = len(sentences)
        
        # Word statistics
        if words:
            word_lengths = [len(w) for w in words]
            avg_word_length = float(np.mean(word_lengths))
            unique_words = len(set(w.lower() for w in words))
            lexical_diversity = unique_words / word_count
        else:
            avg_word_length = 0.0
            unique_words = 0
            lexical_diversity = 0.0
        
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # POS counts
        pos_tags = self.get_pos_tags(text)
        noun_count = pos_tags.get('NOUN', 0) + pos_tags.get('PROPN', 0)
        verb_count = pos_tags.get('VERB', 0)
        adjective_count = pos_tags.get('ADJ', 0)
        
        # Entity count
        entities = self.extract_entities(text)
        entity_count = len(entities)
        
        # Sentiment (simple lexicon-based)
        sentiment_score = self._calculate_sentiment(text)
        
        # Readability
        readability_score = self._calculate_readability(text, sentences, words)
        
        # Punctuation ratios
        exclamation_ratio = text.count('!') / max(char_count, 1)
        question_ratio = text.count('?') / max(char_count, 1)
        capital_ratio = sum(1 for c in text if c.isupper()) / max(char_count, 1)
        digit_ratio = sum(1 for c in text if c.isdigit()) / max(char_count, 1)
        
        return TextFeatures(
            char_count=char_count,
            word_count=word_count,
            sentence_count=sentence_count,
            avg_word_length=avg_word_length,
            avg_sentence_length=avg_sentence_length,
            unique_words=unique_words,
            lexical_diversity=lexical_diversity,
            noun_count=noun_count,
            verb_count=verb_count,
            adjective_count=adjective_count,
            entity_count=entity_count,
            sentiment_score=sentiment_score,
            readability_score=readability_score,
            exclamation_ratio=exclamation_ratio,
            question_ratio=question_ratio,
            capital_ratio=capital_ratio,
            digit_ratio=digit_ratio
        )
    
    def _calculate_sentiment(self, text: str) -> float:
        """
        Simple lexicon-based sentiment analysis
        
        Returns: Sentiment score between -1 and 1
        - Positive values indicate positive sentiment
        - Negative values indicate negative sentiment
        - Zero indicates neutral sentiment
        
        Performance: O(n) where n is word count
        """
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'best',
            'success', 'successful', 'win', 'winning', 'positive', 'growth',
            'increase', 'improve', 'improvement', 'benefit', 'beneficial',
            'breakthrough', 'achievement', 'advance', 'progress', 'gain',
            'strong', 'leading', 'innovative', 'revolutionary', 'record'
        }
        negative_words = {
            'bad', 'poor', 'worst', 'terrible', 'awful', 'fail', 'failure',
            'loss', 'negative', 'decline', 'decrease', 'crisis', 'problem',
            'issue', 'concern', 'risk', 'threat', 'danger', 'disaster',
            'weak', 'down', 'drop', 'fall', 'recession', 'bankruptcy',
            'scandal', 'fraud', 'lawsuit', 'investigation', 'collapse'
        }
        
        words = set(text.lower().split())
        positive_count = len(words & positive_words)
        negative_count = len(words & negative_words)
        total = positive_count + negative_count
        
        if total == 0:
            return 0.0
        
        return (positive_count - negative_count) / total
    
    def _calculate_readability(self, text: str, sentences: List[str], 
                               words: List[str]) -> float:
        """
        Calculate Flesch Reading Ease score
        
        Formula: 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
        
        Score interpretation:
        - 90-100: Very easy (5th grade)
        - 80-90: Easy (6th grade)
        - 70-80: Fairly easy (7th grade)
        - 60-70: Standard (8th-9th grade)
        - 50-60: Fairly difficult (10th-12th grade)
        - 30-50: Difficult (College)
        - 0-30: Very difficult (Graduate)
        
        Returns: Score between 0-100 (higher = easier to read)
        """
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        syllable_count = sum(self._count_syllables(w) for w in words)
        avg_syllables = syllable_count / len(words)
        
        flesch = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        return max(0.0, min(100.0, flesch))
    
    def _count_syllables(self, word: str) -> int:
        """
        Count syllables in a word using rules
        
        Performance: O(n) where n is word length
        """
        word = word.lower()
        
        if len(word) <= 3:
            return 1
        
        vowels = 'aeiouy'
        syllables = 0
        prev_is_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                syllables += 1
            prev_is_vowel = is_vowel
        
        # Adjust for silent e
        if word.endswith('e'):
            syllables -= 1
        
        # Adjust for -le ending
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            syllables += 1
        
        return max(1, syllables)
    
    def preprocess_pipeline(self, text: str, lemmatize: bool = True) -> str:
        """
        Complete preprocessing pipeline for ML classification
        
        Steps:
        1. Clean text (remove URLs, emails, HTML, numbers)
        2. Tokenize (split into words)
        3. Remove stopwords
        4. Lemmatize (convert to base form)
        
        Performance: O(n) overall
        
        Args:
            text: Input text
            lemmatize: Whether to apply lemmatization
            
        Returns:
            Preprocessed text ready for vectorization
        """
        # Clean text
        cleaned = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned, remove_stopwords=True)
        
        # Lemmatize
        if lemmatize:
            tokens = self.lemmatize(tokens)
        
        return ' '.join(tokens)
    
    def get_ngrams(self, tokens: List[str], n: int = 2) -> List[str]:
        """
        Generate n-grams from tokens
        
        N-grams are contiguous sequences of n items from a given sample of text.
        Bigrams (n=2): "new york", "york city"
        Trigrams (n=3): "new york city"
        
        Performance: O(n) where n is number of tokens
        
        Args:
            tokens: List of tokens
            n: N-gram size
            
        Returns:
            List of n-grams
        """
        if len(tokens) < n:
            return []
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def get_keywords_frequency(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords using frequency-based scoring
        
        Performance: O(n log n) due to sorting
        
        Args:
            text: Input text
            top_n: Number of keywords to return
            
        Returns:
            List of (keyword, score) tuples
        """
        tokens = self.tokenize(text, remove_stopwords=True)
        
        if not tokens:
            return []
        
        word_freq = Counter(tokens)
        max_freq = max(word_freq.values())
        
        return [
            (word, count / max_freq)
            for word, count in word_freq.most_common(top_n)
        ]


# Convenience function for quick preprocessing
def preprocess_text(text: str, lemmatize: bool = True) -> str:
    """
    Quick preprocessing function
    
    Args:
        text: Input text
        lemmatize: Whether to apply lemmatization
        
    Returns:
        Preprocessed text
    """
    processor = AdvancedTextProcessor()
    return processor.preprocess_pipeline(text, lemmatize=lemmatize)


if __name__ == '__main__':
    # Demo
    processor = AdvancedTextProcessor()
    
    sample_text = """
    Apple Inc. announced record quarterly earnings of $120 billion, 
    driven by strong iPhone sales and growing Services revenue. 
    CEO Tim Cook highlighted the company's expansion into emerging markets.
    """
    
    print("Original Text:")
    print(sample_text)
    print("\n" + "="*60)
    
    # Preprocess
    processed = processor.preprocess_pipeline(sample_text)
    print("Processed Text:")
    print(processed)
    print("\n" + "="*60)
    
    # Extract features
    features = processor.extract_features(sample_text)
    print("Extracted Features:")
    for field in features.__dataclass_fields__:
        print(f"  {field}: {getattr(features, field)}")
    print("\n" + "="*60)
    
    # Extract entities
    entities = processor.extract_entities(sample_text)
    print("Named Entities:")
    for ent in entities:
        print(f"  {ent['text']} ({ent['label']})")