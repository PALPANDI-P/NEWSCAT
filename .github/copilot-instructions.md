# NEWSCAT - AI Coding Agent Instructions

**Project**: News Classification System with Hybrid ML Ensemble  
**Stack**: Flask backend + Vanilla JS frontend | Scikit-learn classifiers + NLTK text processing

---

## Architecture Overview

### Core Components
- **Backend** (`backend/`): Flask REST API serving classification models
- **Models** (`backend/models/`): Two classifier implementations
  - `SimpleNewsClassifier`: TF-IDF + SVM pipeline (production-ready)
  - `EnhancedNewsClassifier`: Stub class (planned for advanced ML ensemble)
  - `TextProcessor`: Shared text cleaning/preprocessing with NLTK
  - `KeywordExtractor`: NLP feature extraction
  - `BaseNewsClassifier`: Abstract base for all classifiers
- **Frontend** (`frontend/`): Static HTML/CSS/JS with glassmorphism UI
- **Config** (`backend/config.py`): Centralized settings (paths, ML params, categories)

### Data Flow
1. Frontend sends text via `/api/classify` POST
2. Backend validates & routes to `SimpleNewsClassifier` or `EnhancedNewsClassifier`
3. `TextProcessor` cleans input (URLs, emails, special chars, lemmatization)
4. TF-IDF vectorizer + SVM model classifies into 10 categories
5. Response includes category, confidence, sentiment, keywords, readability

### Key Design Decisions
- **Dual Classifier Pattern**: Always maintain both implementations; fallback to `SimpleClassifier` if enhanced unavailable
- **In-Memory Processing**: No data persistence (privacy-first, all processing ephemeral)
- **Config-Driven ML**: ML hyperparameters (TFIDF_MAX_FEATURES, NGRAM_RANGE, etc.) stored in config, not hardcoded
- **Graceful Degradation**: Missing NLTK falls back to basic stopword lists; missing classifiers caught and logged

---

## Critical Developer Workflows

### Running the Application
```bash
# From e:\NEWSCAT_VSCODE root
python backend/app.py          # Starts on http://127.0.0.1:5000
```
- Flask debug mode enabled by default (see `backend/config.py` FLASK_DEBUG)
- Logs go to `logs/newscat.log` + console
- Check startup output for classifier initialization status

### Testing Classification
**Frontend**: Paste text in UI, toggle "Use Enhanced Model", click Analyze  
**API Direct**:
```
POST http://localhost:5000/api/classify
{
  "text": "Apple reported record earnings...",
  "enhanced": false
}
```

### Debugging Model Issues
- Check `get_classifier(use_enhanced=True)` logic in `backend/app.py`—handles fallback to SimpleClassifier
- `classifiers` dict initialized in `init_classifiers()` contains references; None values indicate failures
- NLTK failures logged but don't crash—basic processing continues

---

## Project-Specific Conventions

### Text Processing Pipeline
**Always use `TextProcessor.preprocess_pipeline(text)`** for consistency:
- Lowercase
- URL/email removal (regex)
- HTML tag stripping
- Lemmatization (if NLTK available, falls back to stemming)
- Stopword removal
- Non-alphanumeric char removal

Example (from `simple_classifier.py`):
```python
processed_texts = [self.text_processor.preprocess_pipeline(text)
                   for text in texts]
```

### Response Format
All API responses follow this envelope:
```json
{
  "status": "success|error",
  "timestamp": "ISO-8601",
  "model": "classifier_name",
  "enhanced": boolean,
  "category": "category_name",
  "confidence": 0.0-1.0,
  "code": "error code if status=error"
}
```

### Category Definition
10 fixed categories in `backend/config.py`:
```python
CATEGORIES = {
    'politics': 'Government, elections, policies',
    'sports': 'Sports events, athletes, teams',
    'technology': 'Tech innovations, AI, software',
    # ... 7 more
}
```
**Never hardcode categories**—always fetch from config or API `/categories`.

### Model Persistence
- Trained models saved via `joblib` to `backend/data/models/pretrained/`
- SimpleClassifier attempts `_try_load_model()` on init
- Enhanced classifier is a stub—replace `backend/models/enhanced_classifier.py` with real implementation

---

## Integration Points & Dependencies

### External Libraries
- **Flask 3.0** + CORS: Web framework; CORS configured in `app.config['CORS_ORIGINS']`
- **Scikit-learn 1.3**: TfidfVectorizer, SVC, Pipeline, CalibratedClassifierCV
- **NLTK 3.8**: Tokenization, stopwords, lemmatization (graceful fallback if unavailable)
- **Joblib**: Model serialization
- **python-dotenv**: Environment variable loading (see `.env` handling in `backend/config.py`)

### API Endpoints
- `GET /api/health` - Service status & classifier availability
- `GET /api/categories` - Category list with descriptions
- `GET /api/model/info?enhanced={true|false}` - Classifier metadata
- `POST /api/classify` - Main classification (requires `{text, enhanced}`)
- `GET /` - Serves `frontend/index.html`
- `GET /css/<path>`, `/js/<path>` - Frontend assets

### Cross-Component Communication
Frontend calls backend API via `fetch()` in `frontend/js/main.js`:
- `loadCategories()` → `/api/categories`
- `classifyText(text)` → `/api/classify` (examines model-toggle checkbox)
- Model badge updates from `/api/model/info`

---

## Common Tasks & Patterns

### Adding a New Category
1. Add entry to `CATEGORIES` dict in `backend/config.py`
2. Retrain classifiers with labeled samples
3. Frontend categories auto-populate from `/api/categories` API

### Improving Text Processing
Modify `backend/models/text_processor.py`:
- `clean_text()` - Regex-based cleaning
- `tokenize()` - Word splitting logic
- Advanced techniques in methods like `extract_features()`, `calculate_readability()`

### Extending Classification Features
Add method to `BaseNewsClassifier` abstract class → implement in both `SimpleNewsClassifier` and `EnhancedNewsClassifier`. Example: sentiment analysis, entity extraction already supported (feature flags in config).

### Handling Configuration
Use `app.config['KEY']` in Flask context or pass config dict to model constructors. See `backend/models/simple_classifier.py` for pattern:
```python
self.vectorizer = TfidfVectorizer(
    max_features=self.config.get('TFIDF_MAX_FEATURES', 5000),
    ngram_range=self.config.get('NGRAM_RANGE', (1, 2)),
)
```

---

## Testing & Validation

### Input Validation
Minimum 20 chars, maximum 10,000 chars per `backend/config.py` `MIN_TEXT_LENGTH` / `MAX_TEXT_LENGTH`.  
Checked in `app.py` `validate_text()` function.

### Quick Verification
1. Navigate to `http://localhost:5000`
2. Load sample articles (UI buttons provided)
3. Check `/api/health` for classifier status
4. Tail `logs/newscat.log` during testing

---

## Red Flags & Common Mistakes

1. **Hardcoding categories** - Always use config or API, not magic strings
2. **Skipping text preprocessing** - Direct input to classifier causes poor accuracy
3. **Ignoring fallback logic** - Enhanced classifier may be None; always check `get_classifier()` return
4. **NLTK not available** - Code continues but lemmatization uses stem only; test both paths
5. **Config path issues** - Use `Path` objects and `mkdir(parents=True, exist_ok=True)` (see `backend/config.py`)

---

## Files Reference

| File | Purpose |
|------|---------|
| `backend/app.py` | Flask app, routes, classifier initialization |
| `backend/config.py` | Config classes, ML hyperparams, categories |
| `backend/models/base_classifier.py` | Abstract base for classifiers |
| `backend/models/simple_classifier.py` | TF-IDF + SVM implementation |
| `backend/models/text_processor.py` | Text cleaning & NLP preprocessing |
| `frontend/index.html` | UI layout |
| `frontend/js/main.js` | API calls, event handling |
| `logs/newscat.log` | Application logs |
