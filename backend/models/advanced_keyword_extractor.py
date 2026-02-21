"""
Advanced Keyword Extraction Module
Multiple algorithms for production-grade keyword extraction

Algorithms:
1. TF-IDF based extraction - Statistical importance
2. Position-weighted frequency - Early words get higher weight
3. Named Entity recognition - Extract people, places, organizations
4. Hybrid approach - Combines all methods for best results

Performance: O(n) for most operations
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
from dataclasses import dataclass
import logging

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class Keyword:
    """Keyword with metadata"""
    text: str
    score: float
    position: int
    category: str  # 'noun', 'verb', 'entity', 'phrase'


class AdvancedKeywordExtractor:
    """
    Multi-algorithm keyword extraction for news articles
    
    Features:
    - TF-IDF scoring for statistical importance
    - Named Entity Recognition for proper nouns
    - Position-weighted scoring
    - Hybrid approach combining all methods
    
    Usage:
        extractor = AdvancedKeywordExtractor(method='hybrid')
        keywords = extractor.extract("Apple announced new iPhone...", top_n=10)
        
        # Returns:
        # [{'keyword': 'apple', 'score': 0.95}, {'keyword': 'iphone', 'score': 0.82}, ...]
    """
    
    def __init__(self, method: str = 'hybrid'):
        """
        Initialize keyword extractor
        
        Args:
            method: Extraction method
                - 'tfidf': TF-IDF based scoring
                - 'frequency': Simple frequency counting
                - 'entity': Named Entity Recognition
                - 'position': Position-weighted scoring
                - 'hybrid': Combines all methods (recommended)
        """
        self.method = method
        
        # Initialize spaCy for NER
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load('en_core_web_sm')
                logger.info("spaCy loaded for keyword extraction")
            except OSError:
                self.nlp = None
                logger.warning("spaCy model not found, entity extraction disabled")
        else:
            self.nlp = None
        
        # TF-IDF vectorizer for keyword scoring
        if SKLEARN_AVAILABLE:
            self.tfidf = TfidfVectorizer(
                max_features=100,
                ngram_range=(1, 2),
                stop_words='english',
                sublinear_tf=True,
                token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'
            )
        else:
            self.tfidf = None
        
        # Comprehensive stopwords
        self.stop_words = {
            # Articles and determiners
            'the', 'a', 'an', 'this', 'that', 'these', 'those',
            # Conjunctions
            'and', 'or', 'but', 'nor', 'so', 'yet', 'for',
            # Prepositions
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'again', 'once',
            # Pronouns
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
            # Auxiliary verbs
            'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can',
            # Common adverbs and particles
            'not', 'no', 'so', 'very', 'too', 'just', 'also', 'now',
            'then', 'there', 'here', 'where', 'when', 'why', 'how',
            'all', 'each', 'every', 'both', 'few', 'more', 'most',
            'other', 'some', 'such', 'only', 'own', 'same', 'than',
            # Common adjectives (not informative alone)
            'new', 'old', 'good', 'bad', 'great', 'small', 'large',
            'high', 'low', 'long', 'short', 'big', 'little',
            # Numbers and time
            'one', 'two', 'three', 'first', 'second', 'last', 'next',
            # Other common words
            'said', 'say', 'says', 'told', 'tell', 'tells',
            'according', 'based', 'like', 'well', 'back', 'even',
            'get', 'got', 'make', 'made', 'take', 'took'
        }
        
        logger.info(f"KeywordExtractor initialized with method: {method}")
    
    def extract(self, text: str, top_n: int = 10) -> List[Dict[str, float]]:
        """
        Extract keywords from text
        
        Args:
            text: Input text
            top_n: Number of keywords to return
            
        Returns:
            List of dictionaries with 'keyword' and 'score' keys
        """
        if not text or not text.strip():
            return []
        
        if self.method == 'hybrid':
            return self._hybrid_extraction(text, top_n)
        elif self.method == 'tfidf':
            return self._tfidf_extraction(text, top_n)
        elif self.method == 'frequency':
            return self._frequency_extraction(text, top_n)
        elif self.method == 'entity':
            return self._entity_extraction(text, top_n)
        elif self.method == 'position':
            return self._position_weighted_extraction(text, top_n)
        else:
            return self._frequency_extraction(text, top_n)
    
    def _hybrid_extraction(self, text: str, top_n: int) -> List[Dict[str, float]]:
        """
        Hybrid approach combining multiple methods
        
        Weights:
        - Entity extraction: 40% (named entities are important)
        - Frequency: 30% (common words matter)
        - Position: 30% (early words matter)
        
        Performance: O(n) overall
        """
        keyword_scores: Dict[str, float] = {}
        
        # 1. Frequency-based (weight: 0.3)
        freq_keywords = self._frequency_extraction(text, top_n * 2)
        for kw in freq_keywords:
            keyword_scores[kw['keyword']] = kw['score'] * 0.3
        
        # 2. Entity-based (weight: 0.4)
        entity_keywords = self._entity_extraction(text, top_n)
        for kw in entity_keywords:
            if kw['keyword'] in keyword_scores:
                keyword_scores[kw['keyword']] += kw['score'] * 0.4
            else:
                keyword_scores[kw['keyword']] = kw['score'] * 0.4
        
        # 3. Position-weighted (weight: 0.3)
        position_keywords = self._position_weighted_extraction(text, top_n * 2)
        for kw in position_keywords:
            if kw['keyword'] in keyword_scores:
                keyword_scores[kw['keyword']] += kw['score'] * 0.3
            else:
                keyword_scores[kw['keyword']] = kw['score'] * 0.3
        
        # Sort and return top_n
        sorted_keywords = sorted(
            keyword_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        return [
            {'keyword': kw, 'score': round(score, 4)}
            for kw, score in sorted_keywords
            if score > 0
        ]
    
    def _tfidf_extraction(self, text: str, top_n: int) -> List[Dict[str, float]]:
        """
        TF-IDF based keyword extraction
        
        TF-IDF (Term Frequency-Inverse Document Frequency) scores words
        by their importance in a document relative to a corpus.
        
        For single documents, we use the text itself as the corpus.
        
        Performance: O(n log n) due to sorting
        """
        if not SKLEARN_AVAILABLE or not self.tfidf:
            return self._frequency_extraction(text, top_n)
        
        try:
            # Split text into pseudo-documents for TF-IDF
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            if len(sentences) < 2:
                # Not enough sentences, use frequency
                return self._frequency_extraction(text, top_n)
            
            # Fit and transform
            tfidf_matrix = self.tfidf.fit_transform(sentences)
            scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
            
            # Get feature names
            feature_names = self.tfidf.get_feature_names_out()
            
            # Get top indices
            top_indices = scores.argsort()[-top_n:][::-1]
            
            max_score = max(scores) if scores.max() > 0 else 1
            
            return [
                {
                    'keyword': feature_names[i],
                    'score': float(scores[i] / max_score)
                }
                for i in top_indices
                if scores[i] > 0
            ]
        except Exception as e:
            logger.debug(f"TF-IDF extraction failed: {e}")
            return self._frequency_extraction(text, top_n)
    
    def _frequency_extraction(self, text: str, top_n: int) -> List[Dict[str, float]]:
        """
        Frequency-based keyword extraction
        
        Simple but effective: count word occurrences and normalize.
        
        Performance: O(n) for counting + O(m log m) for sorting
        """
        # Tokenize - extract words with 3+ characters
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter stopwords
        words = [w for w in words if w not in self.stop_words]
        
        if not words:
            return []
        
        # Count frequencies
        word_freq = Counter(words)
        
        max_freq = max(word_freq.values())
        
        return [
            {'keyword': word, 'score': count / max_freq}
            for word, count in word_freq.most_common(top_n)
        ]
    
    def _entity_extraction(self, text: str, top_n: int) -> List[Dict[str, float]]:
        """
        Named Entity based keyword extraction
        
        Uses spaCy NER to extract:
        - PERSON: People's names
        - ORG: Organizations
        - GPE: Countries, cities, states
        - LOC: Non-GPE locations
        - PRODUCT: Products
        - EVENT: Events
        - WORK_OF_ART: Books, songs, etc.
        
        Performance: O(n) with spaCy
        """
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            
            entities = []
            # Entity types to include (exclude dates, times, quantities)
            include_types = {'PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 
                           'WORK_OF_ART', 'FAC', 'NORP', 'LAW'}
            
            for ent in doc.ents:
                if ent.label_ in include_types:
                    # Normalize entity text
                    entity_text = ent.text.lower().strip()
                    # Skip single-character entities
                    if len(entity_text) > 1:
                        entities.append(entity_text)
            
            if not entities:
                return []
            
            # Count and score
            entity_freq = Counter(entities)
            max_freq = max(entity_freq.values())
            
            return [
                {'keyword': entity, 'score': count / max_freq}
                for entity, count in entity_freq.most_common(top_n)
            ]
        except Exception as e:
            logger.debug(f"Entity extraction failed: {e}")
            return []
    
    def _position_weighted_extraction(self, text: str, top_n: int) -> List[Dict[str, float]]:
        """
        Position-weighted keyword extraction
        
        Words appearing earlier in the text get higher scores.
        This is based on the journalistic principle that important
        information appears first (inverted pyramid style).
        
        Formula: score = freq_score * 0.6 + position_score * 0.4
        where position_score = 1 - (position / total_words) * 0.5
        
        Performance: O(n) for processing + O(m log m) for sorting
        """
        # Tokenize with positions
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter stopwords
        words = [w for w in words if w not in self.stop_words]
        
        if not words:
            return []
        
        # Calculate position weights
        total_words = len(words)
        word_first_positions: Dict[str, float] = {}
        
        for i, word in enumerate(words):
            if word not in word_first_positions:
                # Earlier positions get higher weight (1.0 to 0.5)
                position_weight = 1.0 - (i / total_words) * 0.5
                word_first_positions[word] = position_weight
        
        # Count frequencies
        word_freq = Counter(words)
        max_freq = max(word_freq.values())
        
        # Combine frequency and position
        scores = {}
        for word, freq in word_freq.items():
            freq_score = freq / max_freq
            pos_score = word_first_positions[word]
            # Weighted combination
            scores[word] = freq_score * 0.6 + pos_score * 0.4
        
        # Sort and return
        sorted_keywords = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {'keyword': kw, 'score': round(score, 4)}
            for kw, score in sorted_keywords[:top_n]
        ]
    
    def extract_with_context(self, text: str, top_n: int = 10) -> List[Dict[str, any]]:
        """
        Extract keywords with additional context
        
        Returns keywords with:
        - keyword: The extracted keyword
        - score: Importance score (0-1)
        - count: Number of occurrences
        - positions: List of character positions
        - is_entity: Whether it's a named entity
        
        Performance: O(n) overall
        """
        if not text:
            return []
        
        # Get entities
        entities = set()
        if self.nlp:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    entities.add(ent.text.lower())
            except:
                pass
        
        # Tokenize and track positions
        word_positions: Dict[str, List[int]] = {}
        for match in re.finditer(r'\b[a-zA-Z]{3,}\b', text.lower()):
            word = match.group()
            if word not in self.stop_words:
                if word not in word_positions:
                    word_positions[word] = []
                word_positions[word].append(match.start())
        
        if not word_positions:
            return []
        
        # Calculate scores
        max_count = max(len(pos) for pos in word_positions.values())
        
        results = []
        for word, positions in word_positions.items():
            count = len(positions)
            score = count / max_count
            
            # Boost score for entities
            is_entity = word in entities
            if is_entity:
                score = min(score * 1.3, 1.0)
            
            results.append({
                'keyword': word,
                'score': round(score, 4),
                'count': count,
                'first_position': positions[0],
                'is_entity': is_entity
            })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:top_n]
    
    def get_key_phrases(self, text: str, top_n: int = 5) -> List[Dict[str, float]]:
        """
        Extract key phrases (n-grams) from text
        
        Key phrases are multi-word expressions that carry meaning:
        - "machine learning"
        - "stock market"
        - "climate change"
        
        Performance: O(n) for extraction + O(m log m) for sorting
        """
        # Extract bigrams
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        words = [w for w in words if w not in self.stop_words]
        
        if len(words) < 2:
            return []
        
        # Generate bigrams
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words) - 1)]
        
        # Count and score
        bigram_freq = Counter(bigrams)
        
        if not bigram_freq:
            return []
        
        max_freq = max(bigram_freq.values())
        
        return [
            {'phrase': phrase, 'score': count / max_freq}
            for phrase, count in bigram_freq.most_common(top_n)
        ]
    
    def get_info(self) -> Dict[str, any]:
        """Get extractor information"""
        return {
            'name': 'AdvancedKeywordExtractor',
            'method': self.method,
            'spacy_available': SPACY_AVAILABLE,
            'sklearn_available': SKLEARN_AVAILABLE,
            'stopword_count': len(self.stop_words)
        }


# Convenience function
def extract_keywords(text: str, top_n: int = 10, method: str = 'hybrid') -> List[Dict[str, float]]:
    """
    Quick keyword extraction function
    
    Args:
        text: Input text
        top_n: Number of keywords to return
        method: Extraction method
        
    Returns:
        List of keywords with scores
    """
    extractor = AdvancedKeywordExtractor(method=method)
    return extractor.extract(text, top_n=top_n)


if __name__ == '__main__':
    # Demo
    print("="*60)
    print("Advanced Keyword Extractor Demo")
    print("="*60)
    
    sample_text = """
    Apple Inc. announced record quarterly earnings of $120 billion, 
    driven by strong iPhone sales and growing Services revenue. 
    CEO Tim Cook highlighted the company's expansion into emerging markets
    and the success of the new Apple Watch Series 9. The tech giant's stock
    surged 5% following the announcement, making Apple the world's most
    valuable company with a market cap of $3 trillion.
    """
    
    print("\nSample Text:")
    print(sample_text.strip())
    print("\n" + "="*60)
    
    # Test different methods
    methods = ['hybrid', 'frequency', 'entity', 'position']
    
    for method in methods:
        print(f"\n{method.upper()} Method:")
        extractor = AdvancedKeywordExtractor(method=method)
        keywords = extractor.extract(sample_text, top_n=8)
        for kw in keywords:
            print(f"  {kw['keyword']}: {kw['score']:.3f}")
    
    print("\n" + "="*60)
    print("\nWith Context:")
    extractor = AdvancedKeywordExtractor(method='hybrid')
    keywords_with_context = extractor.extract_with_context(sample_text, top_n=5)
    for kw in keywords_with_context:
        print(f"  {kw['keyword']}: score={kw['score']:.3f}, count={kw['count']}, entity={kw['is_entity']}")
    
    print("\n" + "="*60)
    print("\nKey Phrases:")
    phrases = extractor.get_key_phrases(sample_text, top_n=5)
    for phrase in phrases:
        print(f"  {phrase['phrase']}: {phrase['score']:.3f}")
    
    print("\n" + "="*60)