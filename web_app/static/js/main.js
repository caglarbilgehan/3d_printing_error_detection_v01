/**
 * 3D Printer Monitoring System - Main JavaScript
 * Core functionality and utilities
 */

// Sidebar toggle
document.getElementById('toggle-sidebar')?.addEventListener('click', function() {
    document.querySelector('.sidebar').classList.toggle('show');
});

// Language change function
function changeLanguage(lang) {
    fetch(`/set-language/${lang}`)
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                location.reload();
            }
        });
}

// Simulated system stats
setInterval(() => {
    const cpuElem = document.getElementById('cpu-usage');
    const memElem = document.getElementById('memory-usage');
    
    if (cpuElem) {
        cpuElem.textContent = (Math.random() * 30 + 20).toFixed(1);
    }
    if (memElem) {
        memElem.textContent = (Math.random() * 20 + 40).toFixed(1);
    }
}, 3000);

// Performance Settings Modal
function loadPerformanceSettings() {
    fetch('/api/performance')
        .then(r => r.json())
        .then(data => {
            document.getElementById('modal-frame-skip').value = data.frame_skip;
            document.getElementById('modal-frame-skip-value').textContent = data.frame_skip;
            document.getElementById('modal-jpeg-quality').value = data.jpeg_quality;
            document.getElementById('modal-jpeg-quality-value').textContent = data.jpeg_quality;
            document.getElementById('modal-resize-factor').value = Math.round(data.resize_factor * 100);
            document.getElementById('modal-resize-factor-value').textContent = Math.round(data.resize_factor * 100);
        });
}

function applyPerformanceSettings() {
    const frameSkip = parseInt(document.getElementById('modal-frame-skip').value);
    const jpegQuality = parseInt(document.getElementById('modal-jpeg-quality').value);
    const resizeFactor = parseInt(document.getElementById('modal-resize-factor').value) / 100;
    
    fetch('/api/performance', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            frame_skip: frameSkip,
            jpeg_quality: jpegQuality,
            resize_factor: resizeFactor
        })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            alert('Performance settings applied successfully!');
            bootstrap.Modal.getInstance(document.getElementById('performanceModal')).hide();
        }
    });
}

// Slider updates
document.getElementById('modal-frame-skip')?.addEventListener('input', function(e) {
    document.getElementById('modal-frame-skip-value').textContent = e.target.value;
});

document.getElementById('modal-jpeg-quality')?.addEventListener('input', function(e) {
    document.getElementById('modal-jpeg-quality-value').textContent = e.target.value;
});

document.getElementById('modal-resize-factor')?.addEventListener('input', function(e) {
    document.getElementById('modal-resize-factor-value').textContent = e.target.value;
});

// Load performance settings when modal opens
document.getElementById('performanceModal')?.addEventListener('show.bs.modal', loadPerformanceSettings);

// Utility Functions
function formatTime(seconds) {
    if (!seconds) return 'Calculating...';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return `${h}h ${m}m`;
}

function formatBytes(bytes) {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}
