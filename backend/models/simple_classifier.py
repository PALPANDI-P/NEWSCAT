"""
SimpleNewsClassifier — Primary keyword + ML text classifier for NEWSCAT.
Supports 150+ news categories with high-accuracy keyword scoring and optional
scikit-learn model fallback.
"""

import re
import logging
import math
import functools
import nltk
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

from backend.models.taxonomy import TAXONOMY_KEYWORDS

logger = logging.getLogger(__name__)


class SimpleNewsClassifier:
    """
    High-accuracy news topic classifier.
    1. Keyword-frequency scoring against TAXONOMY_KEYWORDS.
    2. Optional pre-trained scikit-learn model (joblib) for ML boosting.
    """

    def __init__(self, name: str = "SimpleNewsClassifier", version: str = "9.0"):
        self.name = name
        self.version = version
        self.model = None  # Optional sklearn model loaded via joblib
        self._keywords = TAXONOMY_KEYWORDS
        self._categories = list(TAXONOMY_KEYWORDS.keys())
        self._word_regex = re.compile(r"\b\w+\b")  # Pre-compiled for speed
        # Ensure Expert NLP NLTK data is present for TextBlob/NounPhrases/Summaries
        try:
            # Vercel's filesystem is read-only. Use /tmp for NLTK data.
            import os
            # Use /tmp/nltk_data for both local (for testing) and Vercel
            # But on local Windows, /tmp doesn't exist, so we use a subfolder in tmp if available
            # For Vercel, it ALWAYS exists.
            home = os.path.expanduser("~")
            nltk_dir = os.path.join("/tmp", "nltk_data") if os.path.exists("/tmp") else os.path.join(home, "nltk_data")
            
            if not os.path.exists(nltk_dir):
                os.makedirs(nltk_dir, exist_ok=True)
            
            if nltk_dir not in nltk.data.path:
                nltk.data.path.append(nltk_dir)
                
            for package in ["punkt", "brown", "averaged_perceptron_tagger", "wordnet"]:
                # Only download if not already there to save time and prevent read-only errors
                nltk.download(package, download_dir=nltk_dir, quiet=True)
        except Exception as e:
            logger.warning(f"NLTK data download failed: {e}")

        logger.info(
            f"{self.name} v{self.version} initialized with "
            f"{len(self._categories)} categories"
        )

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def classify(
        self,
        text: str,
        include_confidence: bool = True,
        include_all_scores: bool = False,
    ) -> Dict[str, Any]:
        """
        Classify news text into the best matching category.

        Returns
        -------
        dict with keys:
            category        : str   – winning category slug
            confidence      : float – 0-100 percentage
            top_predictions : list  – ranked [{category, confidence, category_display}]
            model_name      : str
            model_version   : str
        """
        if not text or len(text.strip()) < 3:
            return self._empty_result()

        text_lower = text.lower().strip()

        # Step 1: keyword scoring (pass original text for title-case detection)
        scores = self._keyword_score(text.strip())

        # Step 2: optional ML model boost
        if self.model is not None:
            scores = self._ml_boost(text, scores)

        # Step 3: rank
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Step 4: normalise top scores to 0-100 confidence
        top_category, top_score = ranked[0] if ranked else ("technology", 0)
        confidence = self._score_to_confidence(top_score, ranked)

        # Step 5: build top predictions
        top_predictions = []
        if include_all_scores:
            for cat, sc in ranked[:10]:
                cat_conf = self._score_to_confidence(sc, ranked)
                top_predictions.append(
                    {
                        "category": cat,
                        "confidence": round(cat_conf, 2),
                        "category_display": cat.replace("_", " ").title(),
                    }
                )

        summary = self._generate_summary(text)

        return {
            "category": top_category,
            "confidence": round(confidence, 2),
            "top_predictions": top_predictions,
            "model_name": self.name,
            "model_version": self.version,
            "category_display": top_category.replace("_", " ").title(),
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # KEYWORD SCORING ENGINE
    # ------------------------------------------------------------------

    @functools.lru_cache(maxsize=1000)
    def _keyword_score(self, text: str) -> Dict[str, float]:
        """Score every category by counting keyword hits with expert heuristics."""
        scores: Dict[str, float] = {}
        text_lower = text.lower()
        
        # Tokenization & N-grams extraction
        words_original = self._word_regex.findall(text)
        words_lower = [w.lower() for w in words_original]
        text_words = set(words_lower)
        
        # Build bigrams and trigrams for fast semantic matching
        bigrams = set(f"{words_lower[i]} {words_lower[i+1]}" for i in range(len(words_lower)-1))
        trigrams = set(f"{words_lower[i]} {words_lower[i+1]} {words_lower[i+2]}" for i in range(len(words_lower)-2))
        
        # TF-IDF Normalization base
        text_len = max(1, len(text_lower))
        length_penalty = math.sqrt(text_len) / 10.0  # dampen scores on massive texts

        # Pre-calculate title case entities (e.g., Apple, NASA)
        title_case_words = set(w.lower() for w in words_original if w.istitle() or w.isupper())
        
        # Expert Semantic Feature: Noun Phrases
        noun_phrases = set()
        if TextBlob:
            try:
                # Extract noun phrases to understand core subjects rather than individual words
                blob = TextBlob(text)
                for np in blob.noun_phrases:
                    noun_phrases.add(np.lower())
            except Exception:
                pass
        # EXPERT ACCURACY BOOST: Title-based weighting
        # We assume the first 400 characters contain the headline/essential context
        title_text = text[:400].lower()
        body_text = text.lower()
        
        # Word counts for density analysis
        words_count = len(self._word_regex.findall(body_text))
        if words_count < 1:
            return {}

        for category, keywords in self._keywords.items():
            cat_score = 0.0
            
            # 1. KEYWORD & N-GRAM MATCHING
            for kw in keywords:
                kw_lower = kw.lower()
                kw_parts = kw_lower.split()
                
                # Dynamic Regex for accurate word boundaries
                pattern = rf"(?i)\b{re.escape(kw_lower)}\b"
                
                # Title Match (4x weight for headlines)
                title_matches = len(re.findall(pattern, title_text))
                cat_score += title_matches * 4.0
                
                # Body Match (1x weight)
                body_matches = len(re.findall(pattern, body_text))
                cat_score += body_matches * 1.2 # Slightly boosted over 1.0
                
                # N-Gram Multiplier (longer phrases like "Artificial Intelligence" are very specific)
                if len(kw_parts) > 1:
                    cat_score += (body_matches + title_matches) * (len(kw_parts) * 1.5)

                # Case Match Bonus (e.g., "AI", "IPO", "GDP")
                if len(kw) <= 4 and kw.isupper():
                    case_match_pattern = rf"\b{re.escape(kw)}\b"
                    if re.search(case_match_pattern, text[:1000]):
                        cat_score += 8.0 
            
            # 2. ADVANCED NLP ENHANCEMENTS (Noun Phrases & Entity Context)
            if noun_phrases:
                # If a keyword is a core noun phrase, boost it significantly
                for kw in keywords:
                    if kw.lower() in noun_phrases:
                        cat_score += 12.0
            
            # 3. GLOBAL HEURISTIC BOOSTS
            if category == "artificial_intelligence" and "gpt" in body_text:
                cat_score += 10.0
            if category == "breaking_news" and any(x in title_text for x in ["breaking", "urgent", "just in", "developing story"]):
                cat_score += 20.0
            if category == "geopolitics" and any(x in body_text for x in ["sanctions", "alliance", "diplomatic", "treaty"]):
                cat_score += 5.0
            
            if cat_score > 0:
                # Final Normalize and Weighting
                density_bonus = (cat_score / words_count) * 50
                scores[category] = cat_score + density_bonus
                
        return scores

    def _score_to_confidence(
        self, score: float, ranked: List[Tuple[str, float]]
    ) -> float:
        """Convert raw score to 0-100 confidence with separation-based scaling."""
        if score <= 0:
            return 5.0

        # Use sigmoid-like scaling — high scores plateau near 100
        base = min(98.0, 45.0 + 25.0 * math.log(score + 1, 2))

        # Boost if there is clear separation from runner-up
        if len(ranked) >= 2:
            runner_up_score = ranked[1][1]
            if runner_up_score > 0:
                gap_ratio = score / (runner_up_score + 0.001)
                if gap_ratio > 3:
                    base = min(99.9, base + 15)
                elif gap_ratio > 2:
                    base = min(99.0, base + 10)
                elif gap_ratio > 1.5:
                    base = min(97.0, base + 5)

        return max(5.0, min(99.9, base))

    # ------------------------------------------------------------------
    # OPTIONAL ML BOOST
    # ------------------------------------------------------------------

    def _ml_boost(self, text: str, scores: Dict[str, float]) -> Dict[str, float]:
        """If a pre-trained sklearn model is loaded, use its predictions to boost scores."""
        try:
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba([text])[0]
                classes = self.model.classes_
                for idx, cls in enumerate(classes):
                    if cls in scores:
                        scores[cls] += proba[idx] * 20  # increased boost factor
            elif hasattr(self.model, "predict"):
                pred = self.model.predict([text])[0]
                if pred in scores:
                    scores[pred] += 10.0
        except Exception as e:
            logger.warning(f"ML boost failed: {e}")
        return scores

    # ------------------------------------------------------------------
    # EXTRACTIVE SUMMARIZATION
    # ------------------------------------------------------------------
    
    def _generate_summary(self, text: str) -> str:
        """Extracts the most relevant 'short form' summary from the text."""
        if not text:
            return "No content available."
        
        # Clean text
        text_clean = re.sub(r'\s+', ' ', text).strip()
        
        # If the text is short enough, just return it
        if len(text_clean) < 150:
            return text_clean
            
        # Try to use TextBlob for intelligent sentence splitting
        best_sentence = ""
        if TextBlob:
            try:
                blob = TextBlob(text_clean)
                sentences = [s.string for s in blob.sentences if len(s.words) > 5]
                if sentences:
                    # Score sentences by length and position (prefer early, mid-length sentences)
                    best_sentence = sentences[0]
                    for s in sentences[:3]:
                        if 15 < len(s.split()) < 40:
                            best_sentence = s
                            break
            except Exception:
                pass
                
        if not best_sentence:
            # Fallback to basic regex sentence split
            sentences = re.split(r'(?<=[.!?])\s+', text_clean)
            valid = [s for s in sentences if 10 < len(s.split()) < 50]
            if valid:
                best_sentence = valid[0]
            else:
                best_sentence = text_clean[:200]
            
        # Format as classification-based summary
        summary = best_sentence.strip()
        if len(summary) > 250:
            summary = summary[:247] + "..."
            
        return summary

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "category": "technology",
            "confidence": 5.0,
            "top_predictions": [],
            "model_name": self.name,
            "model_version": self.version,
            "category_display": "Technology",
        }
