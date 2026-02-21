/**
 * NEWSCAT v4.1 - Optimized Multi-Modal UI JavaScript
 * Ultra-Fast Classification with Caching Support
 */

// ===== CONFIGURATION =====
const API_BASE = '/api';
const MIN_CHARS = 20;
const MAX_FILE_SIZES = {
    image: 10 * 1024 * 1024,
    audio: 50 * 1024 * 1024,
    video: 100 * 1024 * 1024
};

// State
let currentInputType = 'text';
let selectedFiles = { image: null, audio: null, video: null };
let isAnalyzing = false;

// Sample articles
const sampleArticles = {
    tech: "OpenAI has unveiled GPT-5, demonstrating unprecedented natural language capabilities. The model features enhanced reasoning and improved contextual awareness, potentially revolutionizing healthcare and education. Early tests show 40% improvement in complex problem-solving tasks compared to previous versions.",
    sports: "In a stunning Wimbledon upset, unseeded Maria Rodriguez defeated reigning champion Novak Djokovic in straight sets 6-4, 7-5. The 21-year-old Spanish player showcased exceptional athleticism and strategic play, marking the first time in 15 years that a qualifier has reached the quarterfinals.",
    politics: "The Senate passed a landmark $500 billion climate change bill today with bipartisan support, allocating funds for renewable energy initiatives and carbon capture technology. The legislation includes tax incentives for electric vehicles and funding for green infrastructure projects.",
    business: "Apple reported record quarterly earnings of $120 billion driven by strong iPhone sales and growing Services revenue. The company announced a $90 billion stock buyback program and increased its dividend by 10%."
};

// Category styles
const categoryStyles = {
    'technology': { icon: 'fa-microchip', color: '#6366f1' },
    'sports': { icon: 'fa-futbol', color: '#10b981' },
    'politics': { icon: 'fa-landmark', color: '#a855f7' },
    'business': { icon: 'fa-chart-line', color: '#f59e0b' },
    'entertainment': { icon: 'fa-film', color: '#ec4899' },
    'health': { icon: 'fa-heartbeat', color: '#ef4444' },
    'science': { icon: 'fa-flask', color: '#06b6d4' },
    'world': { icon: 'fa-globe', color: '#3b82f6' },
    'education': { icon: 'fa-graduation-cap', color: '#8b5cf6' },
    'environment': { icon: 'fa-leaf', color: '#22c55e' }
};

// ===== LOADING STATE MANAGEMENT =====
function showLoadingOverlay(message = 'Analyzing...', subMessage = 'Please wait while we process your content') {
    const overlay = document.getElementById('loading-overlay');
    const loadingText = document.getElementById('loading-text');
    const loadingSubtext = document.getElementById('loading-subtext');

    if (overlay) {
        if (loadingText) loadingText.textContent = message;
        if (loadingSubtext) loadingSubtext.textContent = subMessage;
        overlay.classList.add('active');
    }
}

function hideLoadingOverlay() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.classList.remove('active');
    }
}

function updateLoadingMessage(message, subMessage = null) {
    const loadingText = document.getElementById('loading-text');
    const loadingSubtext = document.getElementById('loading-subtext');

    if (loadingText) loadingText.textContent = message;
    if (subMessage && loadingSubtext) loadingSubtext.textContent = subMessage;
}

// ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupDragAndDrop();
});

async function initializeApp() {
    await Promise.all([loadCategories(), loadModelStatus()]);
    setupEventListeners();
    updateAnalyzeButton();
}

function setupEventListeners() {
    const textarea = document.getElementById('news-text');
    if (textarea) {
        textarea.addEventListener('input', () => {
            updateCharCount();
            updateAnalyzeButton();
        });
    }
}

// ===== INPUT TYPE SWITCHING =====
function switchInputType(type) {
    currentInputType = type;

    // Update selector buttons
    document.querySelectorAll('.selector-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.type === type);
    });

    // Update panels
    document.querySelectorAll('.input-panel').forEach(panel => {
        panel.classList.remove('active');
    });

    const targetPanel = document.getElementById(`${type}-panel`);
    if (targetPanel) {
        targetPanel.classList.add('active');
    }

    updateAnalyzeButton();
}

// ===== TEXT FUNCTIONS =====
function loadSample(type) {
    const textarea = document.getElementById('news-text');
    if (textarea && sampleArticles[type]) {
        textarea.value = sampleArticles[type];
        updateCharCount();
        updateAnalyzeButton();
    }
}

function clearText() {
    const textarea = document.getElementById('news-text');
    if (textarea) {
        textarea.value = '';
        updateCharCount();
        updateAnalyzeButton();
        hideResults();
    }
}

function updateCharCount() {
    const textarea = document.getElementById('news-text');
    const charCount = document.getElementById('char-count');
    if (textarea && charCount) {
        charCount.textContent = textarea.value.length;
    }
}

// ===== FILE HANDLING =====
function setupDragAndDrop() {
    ['image', 'audio', 'video'].forEach(type => {
        const dropZone = document.getElementById(`${type}-drop-zone`);
        if (dropZone) {
            dropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                dropZone.classList.add('dragover');
            });

            dropZone.addEventListener('dragleave', () => {
                dropZone.classList.remove('dragover');
            });

            dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropZone.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    handleFile(files[0], type);
                }
            });
        }
    });
}

function handleFileSelect(event, type) {
    const file = event.target.files[0];
    if (file) {
        handleFile(file, type);
    }
}

function handleFile(file, type) {
    // Validate file size
    if (file.size > MAX_FILE_SIZES[type]) {
        showNotification(`File too large. Maximum ${MAX_FILE_SIZES[type] / 1024 / 1024}MB allowed.`, 'error');
        return;
    }

    selectedFiles[type] = file;

    if (type === 'image') {
        const reader = new FileReader();
        reader.onload = (e) => {
            const preview = document.getElementById('image-preview');
            const previewImg = document.getElementById('image-preview-img');
            const uploadContent = document.querySelector('#image-drop-zone .upload-content');

            if (preview && previewImg) {
                previewImg.src = e.target.result;
                preview.style.display = 'block';
                uploadContent.style.display = 'none';
            }
        };
        reader.readAsDataURL(file);
    } else {
        const infoEl = document.getElementById(`${type}-info`);
        const filenameEl = document.getElementById(`${type}-filename`);
        const uploadContent = document.querySelector(`#${type}-drop-zone .upload-content`);

        if (infoEl && filenameEl) {
            filenameEl.textContent = file.name;
            infoEl.style.display = 'flex';
            uploadContent.style.display = 'none';
        }
    }

    updateAnalyzeButton();
}

function removeFile(type) {
    selectedFiles[type] = null;

    if (type === 'image') {
        const preview = document.getElementById('image-preview');
        const uploadContent = document.querySelector('#image-drop-zone .upload-content');
        if (preview) preview.style.display = 'none';
        if (uploadContent) uploadContent.style.display = 'block';
    } else {
        const infoEl = document.getElementById(`${type}-info`);
        const uploadContent = document.querySelector(`#${type}-drop-zone .upload-content`);
        if (infoEl) infoEl.style.display = 'none';
        if (uploadContent) uploadContent.style.display = 'block';
    }

    document.getElementById(`${type}-input`).value = '';
    updateAnalyzeButton();
}

// ===== ANALYZE BUTTON =====
function updateAnalyzeButton() {
    const btn = document.getElementById('analyze-btn');
    if (!btn) return;

    let isValid = false;

    switch (currentInputType) {
        case 'text':
            const textarea = document.getElementById('news-text');
            isValid = textarea && textarea.value.trim().length >= MIN_CHARS;
            break;
        case 'image':
        case 'audio':
        case 'video':
            isValid = selectedFiles[currentInputType] !== null;
            break;
    }

    btn.disabled = !isValid || isAnalyzing;
}

// ===== ANALYSIS =====
async function analyzeContent() {
    if (isAnalyzing) return;

    isAnalyzing = true;
    const btn = document.getElementById('analyze-btn');
    btn.classList.add('loading');
    btn.disabled = true;

    // Show loading overlay with appropriate message
    const loadingMessages = {
        'text': { main: 'Analyzing Text...', sub: 'Processing your news article with AI' },
        'image': { main: 'Analyzing Image...', sub: 'Extracting text and classifying content' },
        'audio': { main: 'Analyzing Audio...', sub: 'Transcribing and classifying content' },
        'video': { main: 'Analyzing Video...', sub: 'Extracting frames and classifying content' }
    };

    const messages = loadingMessages[currentInputType] || loadingMessages['text'];
    showLoadingOverlay(messages.main, messages.sub);

    try {
        let result;

        switch (currentInputType) {
            case 'text':
                updateLoadingMessage('Processing text...', 'Running ensemble classification');
                result = await analyzeText();
                break;
            case 'image':
                updateLoadingMessage('Processing image...', 'Extracting visual features');
                result = await analyzeImage();
                break;
            case 'audio':
                updateLoadingMessage('Processing audio...', 'Transcribing speech to text');
                result = await analyzeAudio();
                break;
            case 'video':
                updateLoadingMessage('Processing video...', 'Analyzing video frames');
                result = await analyzeVideo();
                break;
        }

        if (result) {
            updateLoadingMessage('Finalizing results...', 'Almost done!');
            setTimeout(() => {
                displayResults(result);
            }, 300);
        }
    } catch (error) {
        showNotification(`Analysis failed: ${error.message}`, 'error');
    } finally {
        hideLoadingOverlay();
        isAnalyzing = false;
        btn.classList.remove('loading');
        updateAnalyzeButton();
    }
}

async function analyzeText() {
    const textarea = document.getElementById('news-text');
    const text = textarea.value.trim();

    const response = await fetch(`${API_BASE}/classify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, enhanced: true })
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Classification failed');
    }

    return await response.json();
}

async function analyzeImage() {
    const file = selectedFiles.image;
    if (!file) throw new Error('No image selected');

    const formData = new FormData();
    formData.append('image', file);

    const response = await fetch(`${API_BASE}/classify/image`, {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Image classification failed');
    }

    return await response.json();
}

async function analyzeAudio() {
    const file = selectedFiles.audio;
    if (!file) throw new Error('No audio selected');

    const formData = new FormData();
    formData.append('audio', file);

    const response = await fetch(`${API_BASE}/classify/audio`, {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Audio classification failed');
    }

    return await response.json();
}

async function analyzeVideo() {
    const file = selectedFiles.video;
    if (!file) throw new Error('No video selected');

    const formData = new FormData();
    formData.append('video', file);

    const response = await fetch(`${API_BASE}/classify/video`, {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Video classification failed');
    }

    return await response.json();
}

// ===== RESULTS DISPLAY =====
function displayResults(data) {
    const section = document.getElementById('results-section');
    if (!section) return;

    // Show results section
    section.style.display = 'block';
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Main result
    const category = data.category || 'unknown';
    const confidence = (data.confidence || 0) * 100;

    document.getElementById('result-category').textContent = capitalizeFirst(category);
    document.getElementById('result-confidence').textContent = `${confidence.toFixed(1)}%`;

    const confidenceFill = document.getElementById('confidence-fill');
    if (confidenceFill) {
        confidenceFill.style.width = `${confidence}%`;
    }

    // Processing time
    const timeEl = document.getElementById('result-time');
    if (timeEl) {
        const time = data.processing_time_ms || 0;
        timeEl.textContent = time < 1000 ? `${time.toFixed(0)}ms` : `${(time / 1000).toFixed(2)}s`;
    }

    // Model
    const modelEl = document.getElementById('result-model');
    if (modelEl) {
        modelEl.textContent = data.model || 'Unknown';
    }

    // Cache status
    const cacheEl = document.getElementById('result-cache');
    if (cacheEl) {
        cacheEl.textContent = data.cached ? 'Hit' : 'Miss';
        cacheEl.style.color = data.cached ? '#22c55e' : '#f59e0b';
    }

    // Meta
    const metaEl = document.getElementById('result-meta');
    if (metaEl) {
        metaEl.textContent = `Input: ${data.input_type || 'text'}`;
    }

    // Top predictions
    displayPredictions(data.top_predictions || [{ category, confidence: data.confidence }]);

    // Keywords
    if (data.keywords && data.keywords.length > 0) {
        displayKeywords(data.keywords);
    } else {
        document.getElementById('keywords-section').style.display = 'none';
    }

    showNotification('Classification complete!', 'success');
}

function displayPredictions(predictions) {
    const container = document.getElementById('predictions-list');
    if (!container) return;

    container.innerHTML = predictions.map((pred, index) => {
        const style = categoryStyles[pred.category] || { icon: 'fa-tag', color: '#6366f1' };
        const confidence = (pred.confidence || 0) * 100;

        return `
            <div class="prediction-item">
                <span class="prediction-rank">${index + 1}</span>
                <div class="prediction-info">
                    <span class="prediction-category">
                        <i class="fas ${style.icon}" style="color: ${style.color}; margin-right: 8px;"></i>
                        ${capitalizeFirst(pred.category)}
                    </span>
                    <span class="prediction-confidence">${confidence.toFixed(1)}%</span>
                </div>
                <div class="prediction-bar">
                    <div class="prediction-bar-fill" style="width: ${confidence}%;"></div>
                </div>
            </div>
        `;
    }).join('');
}

function displayKeywords(keywords) {
    const section = document.getElementById('keywords-section');
    const container = document.getElementById('keywords-list');

    if (!section || !container) return;

    section.style.display = 'block';
    container.innerHTML = keywords.map(kw =>
        `<span class="keyword-tag">${typeof kw === 'string' ? kw : kw.word || kw}</span>`
    ).join('');
}

function hideResults() {
    const section = document.getElementById('results-section');
    if (section) {
        section.style.display = 'none';
    }
}

// ===== API CALLS =====
async function loadCategories() {
    try {
        const response = await fetch(`${API_BASE}/categories`);
        if (response.ok) {
            const data = await response.json();
            // Categories loaded successfully
        }
    } catch (error) {
        console.error('Failed to load categories:', error);
    }
}

async function loadModelStatus() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        if (response.ok) {
            const data = await response.json();
            updateStatusIndicator(data);
        }
    } catch (error) {
        updateStatusIndicator(null);
    }
}

function updateStatusIndicator(data) {
    const statusPill = document.getElementById('model-status');
    if (!statusPill) return;

    const indicator = statusPill.querySelector('.status-indicator');
    const label = statusPill.querySelector('.status-label');

    if (data && data.status === 'healthy') {
        const hasOptimized = data.classifiers?.optimized;
        const hasEnsemble = data.classifiers?.ensemble;

        if (indicator) {
            indicator.style.background = '#22c55e';
        }
        if (label) {
            label.textContent = hasOptimized ? 'Optimized AI Ready' : 'AI Ready';
        }

        // Update speed indicator
        const speedEl = document.getElementById('speed-indicator');
        if (speedEl) {
            speedEl.textContent = hasOptimized ? '~15ms' : '~30ms';
        }
    } else {
        if (indicator) {
            indicator.style.background = '#ef4444';
        }
        if (label) {
            label.textContent = 'AI Offline';
        }
    }
}

// ===== NOTIFICATIONS =====
function showNotification(message, type = 'info') {
    const container = document.getElementById('notification-container');
    if (!container) return;

    const icons = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-circle',
        info: 'fa-info-circle'
    };

    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <i class="fas ${icons[type]}"></i>
        <span class="notification-message">${message}</span>
    `;

    container.appendChild(notification);

    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateX(100px)';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// ===== UTILITY FUNCTIONS =====
function capitalizeFirst(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1);
}

function showApiInfo() {
    showNotification('API endpoints: /api/classify, /api/keywords, /api/health', 'info');
}

function showModelInfo() {
    showNotification('Using Optimized Ensemble Classifier v2.1', 'info');
}
