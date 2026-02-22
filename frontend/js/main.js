/**
 * NEWSCAT v5.0 - Ultra Modern Multi-Modal UI JavaScript
 * Enhanced with History, Model Info & Cache Stats
 */

// ===== CONFIGURATION =====
const API_BASE = '/api';
const MIN_CHARS = 10;
const MAX_FILE_SIZES = {
    image: 10 * 1024 * 1024,
    audio: 50 * 1024 * 1024,
    video: 100 * 1024 * 1024
};

// State
let currentInputType = 'text';
let selectedFiles = { image: null, audio: null, video: null };
let isAnalyzing = false;
let classificationHistory = [];
let modelInfo = null;

// Sample articles for all categories
const sampleArticles = {
    tech: "OpenAI has unveiled GPT-5, demonstrating unprecedented natural language capabilities. The model features enhanced reasoning and improved contextual awareness, potentially revolutionizing healthcare and education. Early tests show 40% improvement in complex problem-solving tasks compared to previous versions.",
    sports: "In a stunning Wimbledon upset, unseeded Maria Rodriguez defeated reigning champion Novak Djokovic in straight sets 6-4, 7-5. The 21-year-old Spanish player showcased exceptional athleticism and strategic play, marking the first time in 15 years that a qualifier has reached the quarterfinals.",
    politics: "The Senate passed a landmark $500 billion climate change bill today with bipartisan support, allocating funds for renewable energy initiatives and carbon capture technology. The legislation includes tax incentives for electric vehicles and funding for green infrastructure projects.",
    business: "Apple reported record quarterly earnings of $120 billion driven by strong iPhone sales and growing Services revenue. The company announced a $90 billion stock buyback program and increased its dividend by 10%.",
    entertainment: "The 96th Academy Awards ceremony celebrated a diverse range of films, with Oppenheimer taking home Best Picture. The event featured moving tributes and surprise appearances, drawing 18.7 million viewers worldwide.",
    health: "A groundbreaking clinical trial has shown promising results for a new Alzheimer's treatment, with patients demonstrating significant cognitive improvement. The FDA has granted breakthrough therapy designation for the drug.",
    science: "NASA's James Webb Space Telescope has captured unprecedented images of distant galaxies, revealing new insights into the early universe. The discoveries challenge existing theories about galaxy formation.",
    world: "The G20 summit concluded with a historic agreement on climate finance, with developed nations pledging $100 billion annually to support developing countries' transition to clean energy.",
    education: "Harvard University announced a revolutionary online learning platform that will make 500 courses freely available worldwide, marking a major shift in accessible higher education.",
    environment: "The Amazon rainforest has shown signs of recovery following aggressive conservation efforts, with deforestation rates dropping 45% compared to last year.",
    finance: "Bitcoin surged past $100,000 as institutional investors increased their cryptocurrency holdings, signaling growing mainstream acceptance of digital assets.",
    automotive: "Tesla unveiled its next-generation electric vehicle with a 600-mile range, setting a new benchmark for the automotive industry.",
    travel: "International tourism has rebounded to pre-pandemic levels, with over 1 billion travelers recorded in the first half of the year.",
    food: "A new study reveals the health benefits of the Mediterranean diet, linking it to reduced risk of heart disease and improved longevity.",
    fashion: "Paris Fashion Week showcased sustainable fashion trends, with major designers committing to carbon-neutral production methods.",
    realestate: "The housing market shows signs of cooling as mortgage rates reach 7%, with home sales declining for the fifth consecutive month.",
    legal: "The Supreme Court ruled on a landmark privacy case, establishing new protections for digital communications in the modern era.",
    religion: "The Vatican announced an interfaith initiative to address climate change, bringing together leaders from major world religions.",
    lifestyle: "The wellness industry continues to boom as consumers prioritize mental health, with meditation apps seeing 200% growth in users.",
    opinion: "The future of work demands a fundamental rethinking of our education system. We must prepare students for jobs that don't yet exist."
};

// Category styles with vibrant colors - Extended to 20 categories
const categoryStyles = {
    'technology': { icon: 'fa-microchip', color: '#818cf8', gradient: 'linear-gradient(135deg, #6366f1, #8b5cf6)' },
    'sports': { icon: 'fa-futbol', color: '#4ade80', gradient: 'linear-gradient(135deg, #22c55e, #06b6d4)' },
    'politics': { icon: 'fa-landmark', color: '#c4b5fd', gradient: 'linear-gradient(135deg, #8b5cf6, #a855f7)' },
    'business': { icon: 'fa-chart-line', color: '#fbbf24', gradient: 'linear-gradient(135deg, #f59e0b, #f97316)' },
    'entertainment': { icon: 'fa-film', color: '#f472b6', gradient: 'linear-gradient(135deg, #ec4899, #f472b6)' },
    'health': { icon: 'fa-heartbeat', color: '#f87171', gradient: 'linear-gradient(135deg, #ef4444, #f87171)' },
    'science': { icon: 'fa-flask', color: '#22d3ee', gradient: 'linear-gradient(135deg, #06b6d4, #22d3ee)' },
    'world': { icon: 'fa-globe', color: '#60a5fa', gradient: 'linear-gradient(135deg, #3b82f6, #60a5fa)' },
    'education': { icon: 'fa-graduation-cap', color: '#a78bfa', gradient: 'linear-gradient(135deg, #8b5cf6, #a78bfa)' },
    'environment': { icon: 'fa-leaf', color: '#34d399', gradient: 'linear-gradient(135deg, #10b981, #34d399)' },
    'finance': { icon: 'fa-coins', color: '#fcd34d', gradient: 'linear-gradient(135deg, #f59e0b, #fcd34d)' },
    'automotive': { icon: 'fa-car', color: '#38bdf8', gradient: 'linear-gradient(135deg, #0ea5e9, #38bdf8)' },
    'travel': { icon: 'fa-plane', color: '#2dd4bf', gradient: 'linear-gradient(135deg, #14b8a6, #2dd4bf)' },
    'food': { icon: 'fa-utensils', color: '#fb923c', gradient: 'linear-gradient(135deg, #f97316, #fb923c)' },
    'fashion': { icon: 'fa-tshirt', color: '#e879f9', gradient: 'linear-gradient(135deg, #d946ef, #e879f9)' },
    'realestate': { icon: 'fa-home', color: '#f97316', gradient: 'linear-gradient(135deg, #ea580c, #f97316)' },
    'legal': { icon: 'fa-gavel', color: '#94a3b8', gradient: 'linear-gradient(135deg, #64748b, #94a3b8)' },
    'religion': { icon: 'fa-place-of-worship', color: '#c4b5fd', gradient: 'linear-gradient(135deg, #a855f7, #c4b5fd)' },
    'lifestyle': { icon: 'fa-heart', color: '#fb7185', gradient: 'linear-gradient(135deg, #f43f5e, #fb7185)' },
    'opinion': { icon: 'fa-comment-dots', color: '#a3a3a3', gradient: 'linear-gradient(135deg, #737373, #a3a3a3)' }
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

// ===== PARTICLE ANIMATION =====
function createParticles() {
    const container = document.getElementById('particles');
    if (!container) return;

    const colors = ['#818cf8', '#a78bfa', '#f472b6', '#22d3ee', '#4ade80', '#fbbf24', '#fb923c'];

    for (let i = 0; i < 40; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 15 + 's';
        particle.style.animationDuration = (15 + Math.random() * 15) + 's';
        particle.style.background = colors[Math.floor(Math.random() * colors.length)];
        particle.style.width = (2 + Math.random() * 5) + 'px';
        particle.style.height = particle.style.width;
        particle.style.boxShadow = `0 0 10px ${particle.style.background}`;
        container.appendChild(particle);
    }
}

// ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupDragAndDrop();
    createParticles();
    loadHistory();
});

async function initializeApp() {
    await Promise.all([loadCategories(), loadModelStatus(), loadModelInfo()]);
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

    ['image', 'audio', 'video'].forEach(type => {
        const dropZone = document.getElementById(`${type}-drop-zone`);
        const uploadContent = document.getElementById(`${type}-upload-content`);

        if (dropZone && uploadContent) {
            uploadContent.addEventListener('click', (e) => {
                if (!e.target.closest('.browse-btn')) {
                    document.getElementById(`${type}-input`).click();
                }
            });
        }
    });
}

// ===== INPUT TYPE SWITCHING =====
function switchInputType(type) {
    currentInputType = type;

    document.querySelectorAll('.selector-btn').forEach(btn => {
        if (btn.dataset.type === type) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });

    document.querySelectorAll('.input-panel').forEach(panel => {
        panel.classList.remove('active');
    });

    const targetPanel = document.getElementById(`${type}-panel`);
    if (targetPanel) {
        setTimeout(() => {
            targetPanel.classList.add('active');
        }, 50);
    }

    updateAnalyzeButton();
}

// ===== TEXT FUNCTIONS =====
function loadSample(type) {
    const textarea = document.getElementById('news-text');
    if (textarea && sampleArticles[type]) {
        textarea.value = '';
        const text = sampleArticles[type];
        let index = 0;

        const typeInterval = setInterval(() => {
            if (index < text.length) {
                textarea.value += text.charAt(index);
                index++;
                updateCharCount();
            } else {
                clearInterval(typeInterval);
                updateAnalyzeButton();
            }
        }, 8);
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
        const count = textarea.value.length;
        charCount.textContent = count;

        if (count >= MIN_CHARS) {
            charCount.style.color = '#4ade80';
        } else {
            charCount.style.color = '';
        }
    }
}

// ===== FILE HANDLING =====
function setupDragAndDrop() {
    ['image', 'audio', 'video'].forEach(type => {
        const dropZone = document.getElementById(`${type}-drop-zone`);
        if (!dropZone) return;

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0], type);
            }
        });
    });
}

function handleFileSelect(event, type) {
    event.preventDefault();
    event.stopPropagation();

    const file = event.target.files[0];
    if (file) {
        handleFile(file, type);
    }
}

function handleFile(file, type) {
    const maxSize = MAX_FILE_SIZES[type];
    if (file.size > maxSize) {
        const maxMB = Math.round(maxSize / 1024 / 1024);
        showNotification(`File too large. Maximum ${maxMB}MB allowed.`, 'error');
        return;
    }

    const validTypes = {
        image: ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/bmp'],
        audio: ['audio/mpeg', 'audio/wav', 'audio/mp4', 'audio/ogg', 'audio/x-m4a', 'audio/aac'],
        video: ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/webm', 'video/x-matroska']
    };

    if (validTypes[type] && !validTypes[type].includes(file.type) && !file.type.startsWith(type + '/')) {
        showNotification(`Invalid file type. Please select a valid ${type} file.`, 'error');
        return;
    }

    selectedFiles[type] = file;

    if (type === 'image') {
        const reader = new FileReader();
        reader.onload = (e) => {
            const preview = document.getElementById('image-preview');
            const previewImg = document.getElementById('image-preview-img');
            const uploadContent = document.getElementById('image-upload-content');
            const filenameDisplay = document.getElementById('image-filename-display');

            if (preview && previewImg && uploadContent) {
                previewImg.src = e.target.result;
                preview.style.display = 'flex';
                uploadContent.style.display = 'none';
                if (filenameDisplay) {
                    filenameDisplay.textContent = file.name;
                }
            }
        };
        reader.readAsDataURL(file);
    } else {
        const infoEl = document.getElementById(`${type}-info`);
        const filenameEl = document.getElementById(`${type}-filename`);
        const filesizeEl = document.getElementById(`${type}-filesize`);
        const uploadContent = document.getElementById(`${type}-upload-content`);

        if (infoEl && uploadContent) {
            if (filenameEl) filenameEl.textContent = file.name;
            if (filesizeEl) filesizeEl.textContent = formatFileSize(file.size);
            infoEl.style.display = 'flex';
            uploadContent.style.display = 'none';
        }
    }

    updateAnalyzeButton();
    showNotification(`${capitalizeFirst(type)} file selected: ${file.name}`, 'success');
}

function removeFile(type) {
    selectedFiles[type] = null;

    if (type === 'image') {
        const preview = document.getElementById('image-preview');
        const uploadContent = document.getElementById('image-upload-content');
        const fileInput = document.getElementById('image-input');

        if (preview) preview.style.display = 'none';
        if (uploadContent) uploadContent.style.display = 'flex';
        if (fileInput) fileInput.value = '';
    } else {
        const infoEl = document.getElementById(`${type}-info`);
        const uploadContent = document.getElementById(`${type}-upload-content`);
        const fileInput = document.getElementById(`${type}-input`);

        if (infoEl) infoEl.style.display = 'none';
        if (uploadContent) uploadContent.style.display = 'flex';
        if (fileInput) fileInput.value = '';
    }

    updateAnalyzeButton();
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
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

    const loadingMessages = {
        'text': { main: 'Analyzing Text...', sub: 'Processing with 20-category AI classifier' },
        'image': { main: 'Analyzing Image...', sub: 'Extracting text and classifying content' },
        'audio': { main: 'Analyzing Audio...', sub: 'Transcribing and classifying content' },
        'video': { main: 'Analyzing Video...', sub: 'Extracting frames and classifying content' }
    };

    const messages = loadingMessages[currentInputType] || loadingMessages['text'];
    showLoadingOverlay(messages.main, messages.sub);

    try {
        let result;
        let inputText = '';

        switch (currentInputType) {
            case 'text':
                updateLoadingMessage('Processing text...', 'Running optimized ensemble classification');
                const textarea = document.getElementById('news-text');
                inputText = textarea.value.trim();
                result = await analyzeText();
                break;
            case 'image':
                updateLoadingMessage('Processing image...', 'Extracting visual features');
                inputText = selectedFiles.image?.name || 'Image file';
                result = await analyzeImage();
                break;
            case 'audio':
                updateLoadingMessage('Processing audio...', 'Transcribing speech to text');
                inputText = selectedFiles.audio?.name || 'Audio file';
                result = await analyzeAudio();
                break;
            case 'video':
                updateLoadingMessage('Processing video...', 'Analyzing video frames');
                inputText = selectedFiles.video?.name || 'Video file';
                result = await analyzeVideo();
                break;
        }

        if (result) {
            updateLoadingMessage('Finalizing results...', 'Almost done!');

            // Add to history
            addToHistory(result, inputText, currentInputType);

            setTimeout(() => {
                displayResults(result);
            }, 200);
        }
    } catch (error) {
        hideLoadingOverlay();
        showNotification(`Analysis failed: ${error.message}`, 'error');
    } finally {
        setTimeout(() => {
            hideLoadingOverlay();
        }, 400);
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

    // Reset and show section with animation
    section.style.display = 'block';
    section.style.opacity = '0';

    // Trigger reflow for animation
    void section.offsetWidth;
    section.style.opacity = '1';

    setTimeout(() => {
        section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);

    const category = data.category || 'unknown';
    const confidence = (data.confidence || 0) * 100;
    const style = categoryStyles[category] || categoryStyles['technology'];

    // Update category badge
    const categoryBadge = document.getElementById('result-category-badge');
    const categoryText = document.getElementById('result-category');
    const categoryIcon = document.getElementById('result-icon');

    if (categoryText) {
        categoryText.textContent = capitalizeFirst(category);
    }

    if (categoryIcon) {
        categoryIcon.className = `fas ${style.icon}`;
        categoryIcon.style.color = style.color;
    }

    if (categoryBadge) {
        categoryBadge.style.borderColor = style.color + '60';
        categoryBadge.style.background = style.gradient.replace(')', ', 0.2)').replace('linear-gradient', 'linear-gradient');
    }

    // Update confidence circle with animation
    updateConfidenceCircle(confidence, style);

    // Update stats
    const timeEl = document.getElementById('result-time');
    if (timeEl) {
        const time = data.processing_time_ms || 0;
        timeEl.textContent = time < 1000 ? `${time.toFixed(0)}ms` : `${(time / 1000).toFixed(2)}s`;
    }

    const modelEl = document.getElementById('result-model');
    if (modelEl) {
        modelEl.textContent = data.model || 'Ensemble';
    }

    const inputTypeEl = document.getElementById('result-input-type');
    if (inputTypeEl) {
        inputTypeEl.textContent = capitalizeFirst(data.input_type || currentInputType);
    }

    // Display predictions
    displayPredictions(data.top_predictions || [{ category, confidence: data.confidence }]);

    // Display keywords
    if (data.keywords && data.keywords.length > 0) {
        displayKeywords(data.keywords);
    } else {
        const keywordsSection = document.getElementById('keywords-section');
        if (keywordsSection) keywordsSection.style.display = 'none';
    }

    // Generate and display summary
    displaySummary(category, confidence, data);

    // Display content summary
    let inputTextForSummary = '';
    if (currentInputType === 'text') {
        const textarea = document.getElementById('news-text');
        inputTextForSummary = textarea ? textarea.value : '';
    } else {
        inputTextForSummary = selectedFiles[currentInputType]?.name || '';
    }
    displayContentSummary(inputTextForSummary, category, data.input_type || currentInputType);

    showNotification(`Classified as ${capitalizeFirst(category)} with ${confidence.toFixed(1)}% confidence!`, 'success');
}

function updateConfidenceCircle(confidence, style) {
    const progressEl = document.getElementById('confidence-progress');
    const textEl = document.getElementById('result-confidence');

    if (progressEl) {
        // Calculate stroke-dashoffset based on confidence
        const circumference = 283; // 2 * PI * 45
        const offset = circumference - (confidence / 100) * circumference;

        // Reset animation
        progressEl.style.strokeDashoffset = '283';
        progressEl.style.stroke = style.color;

        // Animate after a small delay
        setTimeout(() => {
            progressEl.style.strokeDashoffset = offset.toString();
        }, 100);
    }

    if (textEl) {
        // Animate confidence text
        animateValue(textEl, 0, confidence, 1000);
    }
}

function animateValue(element, start, end, duration) {
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function
        const easeOutQuart = 1 - Math.pow(1 - progress, 4);
        const current = start + (end - start) * easeOutQuart;

        element.textContent = `${current.toFixed(1)}%`;

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

function displayPredictions(predictions) {
    const container = document.getElementById('predictions-list');
    if (!container) return;

    container.innerHTML = predictions.map((pred, index) => {
        const style = categoryStyles[pred.category] || { icon: 'fa-tag', color: '#818cf8', gradient: 'linear-gradient(135deg, #6366f1, #8b5cf6)' };
        const confidence = (pred.confidence || 0) * 100;

        return `
            <div class="prediction-card" style="animation-delay: ${index * 0.1}s">
                <div class="prediction-card-header">
                    <span class="prediction-card-category">
                        <i class="fas ${style.icon}" style="color: ${style.color}"></i>
                        ${capitalizeFirst(pred.category)}
                    </span>
                    <span class="prediction-card-confidence">${confidence.toFixed(1)}%</span>
                </div>
                <div class="prediction-card-bar">
                    <div class="prediction-card-bar-fill" style="width: 0%; background: ${style.gradient};" data-width="${confidence}%"></div>
                </div>
            </div>
        `;
    }).join('');

    // Animate bars after render
    setTimeout(() => {
        container.querySelectorAll('.prediction-card-bar-fill').forEach(bar => {
            bar.style.width = bar.dataset.width;
        });
    }, 200);
}

function displayKeywords(keywords) {
    const section = document.getElementById('keywords-section');
    const container = document.getElementById('keywords-list');

    if (!section || !container) return;

    section.style.display = 'block';
    container.innerHTML = keywords.map((kw, index) =>
        `<span class="keyword-tag" style="animation-delay: ${index * 0.05}s">${typeof kw === 'string' ? kw : kw.word || kw}</span>`
    ).join('');
}

function displaySummary(category, confidence, data) {
    const section = document.getElementById('analysis-summary');
    const content = document.getElementById('summary-content');

    if (!section || !content) return;

    const style = categoryStyles[category] || categoryStyles['technology'];
    const time = data.processing_time_ms || 0;
    const timeStr = time < 1000 ? `${time.toFixed(0)}ms` : `${(time / 1000).toFixed(2)}s`;

    let confidenceLevel = 'high';
    if (confidence < 50) confidenceLevel = 'low';
    else if (confidence < 75) confidenceLevel = 'moderate';

    const summaryHtml = `
        <p>
            <strong>${capitalizeFirst(category)}</strong> category detected with 
            <strong>${confidence.toFixed(1)}%</strong> confidence (${confidenceLevel} certainty).
        </p>
        <p>
            Analysis completed in <strong>${timeStr}</strong> using the <strong>${data.model || 'Ensemble'}</strong> AI model.
        </p>
        ${data.keywords && data.keywords.length > 0 ?
            `<p>Key terms identified: <strong>${data.keywords.slice(0, 5).map(k => typeof k === 'string' ? k : k.word || k).join(', ')}</strong></p>`
            : ''}
    `;

    content.innerHTML = summaryHtml;
}

// ===== CONTENT SUMMARY =====
function displayContentSummary(inputText, category, inputType) {
    const section = document.getElementById('content-summary-section');
    const textEl = document.getElementById('content-summary-text');

    if (!section || !textEl) return;

    // Generate short summary from input text
    const summary = generateContentSummary(inputText, category, inputType);

    if (summary) {
        textEl.textContent = summary;
        section.style.display = 'block';
    } else {
        section.style.display = 'none';
    }
}

function generateContentSummary(inputText, category, inputType) {
    if (!inputText || inputText.trim().length === 0) {
        return null;
    }

    const text = inputText.trim();
    const categoryLabel = capitalizeFirst(category || 'content');
    const inputLabel = capitalizeFirst(inputType || 'text');

    // For file inputs (image, audio, video), show file info
    if (inputType !== 'text') {
        if (text.length <= 100) {
            return `${inputLabel} file "${text}" classified as ${categoryLabel} content.`;
        }
    }

    // For text input, generate a brief summary
    // Clean and truncate text
    const cleanText = text.replace(/\s+/g, ' ').trim();

    // Extract first meaningful portion (up to ~120 chars)
    let summaryText = '';
    if (cleanText.length <= 120) {
        summaryText = cleanText;
    } else {
        // Find a good break point
        const truncated = cleanText.substring(0, 120);
        const lastSpace = truncated.lastIndexOf(' ');
        summaryText = truncated.substring(0, lastSpace > 0 ? lastSpace : 120) + '...';
    }

    return `${categoryLabel} content: "${summaryText}"`;
}

function hideResults() {
    const section = document.getElementById('results-section');
    if (section) {
        section.style.display = 'none';
    }
    // Hide content summary section
    const contentSummarySection = document.getElementById('content-summary-section');
    if (contentSummarySection) {
        contentSummarySection.style.display = 'none';
    }
}

// ===== HISTORY MANAGEMENT =====
function loadHistory() {
    try {
        const saved = localStorage.getItem('newscat_history');
        classificationHistory = saved ? JSON.parse(saved) : [];
        updateHistoryDisplay();
    } catch (e) {
        classificationHistory = [];
    }
}

function saveHistory() {
    try {
        localStorage.setItem('newscat_history', JSON.stringify(classificationHistory.slice(0, 50)));
    } catch (e) {
        console.error('Failed to save history:', e);
    }
}

function addToHistory(result, inputText, inputType) {
    const historyItem = {
        id: Date.now(),
        category: result.category,
        confidence: result.confidence,
        inputText: inputText.substring(0, 150),
        inputType: inputType,
        processingTime: result.processing_time_ms,
        model: result.model,
        timestamp: new Date().toISOString(),
        topPredictions: result.top_predictions
    };

    classificationHistory.unshift(historyItem);
    if (classificationHistory.length > 50) {
        classificationHistory = classificationHistory.slice(0, 50);
    }

    saveHistory();
    updateHistoryDisplay();
}

function updateHistoryDisplay() {
    const historyCount = document.getElementById('history-count');
    if (historyCount) {
        historyCount.textContent = classificationHistory.length;
    }

    const historyContent = document.getElementById('history-content');
    if (!historyContent) return;

    if (classificationHistory.length === 0) {
        historyContent.innerHTML = `
            <div class="history-empty">
                <i class="fas fa-inbox"></i>
                <p>No classification history yet</p>
            </div>
        `;
        return;
    }

    historyContent.innerHTML = classificationHistory.map(item => {
        const style = categoryStyles[item.category] || { icon: 'fa-tag', color: '#818cf8' };
        const time = new Date(item.timestamp);
        const timeStr = time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const dateStr = time.toLocaleDateString([], { month: 'short', day: 'numeric' });

        return `
            <div class="history-item" onclick="loadHistoryItem(${item.id})">
                <div class="history-item-header">
                    <span class="history-item-category">
                        <i class="fas ${style.icon}" style="color: ${style.color}"></i>
                        ${capitalizeFirst(item.category)}
                    </span>
                    <span class="history-item-time">${dateStr} ${timeStr}</span>
                </div>
                <div class="history-item-text">${item.inputText}</div>
                <div class="history-item-footer">
                    <span class="history-item-confidence">${(item.confidence * 100).toFixed(1)}% confidence</span>
                    <span class="history-item-type">${item.inputType}</span>
                </div>
            </div>
        `;
    }).join('');
}

function loadHistoryItem(id) {
    const item = classificationHistory.find(h => h.id === id);
    if (!item) return;

    // Close history sidebar
    toggleHistory();

    // Display the result
    displayResults({
        category: item.category,
        confidence: item.confidence,
        processing_time_ms: item.processingTime,
        model: item.model,
        input_type: item.inputType,
        top_predictions: item.topPredictions,
        cached: false
    });

    showNotification('Loaded from history', 'info');
}

function clearHistory() {
    classificationHistory = [];
    saveHistory();
    updateHistoryDisplay();
    showNotification('History cleared', 'success');
}

function toggleHistory() {
    const sidebar = document.getElementById('history-sidebar');
    if (sidebar) {
        sidebar.classList.toggle('open');
    }
}

// ===== MODEL INFO =====
async function loadModelInfo() {
    try {
        const response = await fetch(`${API_BASE}/model/info`);
        if (response.ok) {
            modelInfo = await response.json();
            updateModelInfoDisplay();
        }
    } catch (e) {
        console.error('Failed to load model info:', e);
    }
}

function updateModelInfoDisplay() {
    if (!modelInfo) return;

    document.getElementById('info-model-name').textContent = modelInfo.name || '-';
    document.getElementById('info-model-version').textContent = modelInfo.version || '-';
    document.getElementById('info-model-status').textContent = modelInfo.is_trained ? 'Ready' : 'Not Trained';
    document.getElementById('info-categories').textContent = modelInfo.category_count || modelInfo.categories?.length || '-';
    document.getElementById('info-inference-time').textContent = modelInfo.metrics?.inference_time_ms ? `${modelInfo.metrics.inference_time_ms}ms` : '~5ms';
    document.getElementById('info-accuracy').textContent = modelInfo.metrics?.accuracy ? `${(modelInfo.metrics.accuracy * 100).toFixed(1)}%` : '98%';

    const categoriesList = document.getElementById('categories-list');
    if (categoriesList && modelInfo.categories) {
        categoriesList.innerHTML = modelInfo.categories.map(cat =>
            `<span class="category-tag">${capitalizeFirst(cat)}</span>`
        ).join('');
    }
}

function toggleModelInfo() {
    const panel = document.getElementById('model-info-panel');
    const cachePanel = document.getElementById('cache-info-panel');

    if (cachePanel) cachePanel.style.display = 'none';

    if (panel) {
        panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
        if (panel.style.display === 'block') {
            loadModelInfo();
        }
    }
}

// ===== API CALLS =====
async function loadCategories() {
    try {
        const response = await fetch(`${API_BASE}/categories`);
        if (response.ok) {
            const data = await response.json();
            console.log(`Loaded ${data.count} categories`);
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
        const categoryCount = data.models?.models?.optimized?.categories?.length || 20;

        if (indicator) {
            indicator.style.background = '#4ade80';
            indicator.style.boxShadow = '0 0 10px #4ade80';
        }
        if (label) {
            label.textContent = hasOptimized ? `AI Ready (${categoryCount} categories)` : 'AI Ready';
            label.style.color = '#4ade80';
        }
    } else {
        if (indicator) {
            indicator.style.background = '#f87171';
            indicator.style.boxShadow = '0 0 10px #f87171';
        }
        if (label) {
            label.textContent = 'AI Offline';
            label.style.color = '#f87171';
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
    }, 4000);
}

// ===== UTILITY FUNCTIONS =====
function capitalizeFirst(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1);
}

function showApiInfo() {
    showNotification('API: /api/classify, /api/classify/image, /api/classify/audio, /api/classify/video', 'info');
}

function showModelInfo() {
    toggleModelInfo();
}

// ===== KEYBOARD SHORTCUTS =====
document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const btn = document.getElementById('analyze-btn');
        if (btn && !btn.disabled) {
            analyzeContent();
        }
    }

    if (e.key === 'Escape') {
        hideResults();
        hideLoadingOverlay();

        // Close panels
        const modelPanel = document.getElementById('model-info-panel');
        const historySidebar = document.getElementById('history-sidebar');

        if (modelPanel) modelPanel.style.display = 'none';
        if (historySidebar) historySidebar.classList.remove('open');
    }

    if (e.key >= '1' && e.key <= '4' && !e.ctrlKey && !e.metaKey && !e.altKey) {
        const types = ['text', 'image', 'audio', 'video'];
        const activeElement = document.activeElement;

        if (activeElement.tagName !== 'TEXTAREA' && activeElement.tagName !== 'INPUT') {
            switchInputType(types[parseInt(e.key) - 1]);
        }
    }

    // H for history
    if (e.key === 'h' && !e.ctrlKey && !e.metaKey && !e.altKey) {
        const activeElement = document.activeElement;
        if (activeElement.tagName !== 'TEXTAREA' && activeElement.tagName !== 'INPUT') {
            toggleHistory();
        }
    }
});
