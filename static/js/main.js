/**
 * Main JavaScript for Automatic Face Grouping Application
 * Handles file upload, drag-and-drop, and AJAX processing
 */

// Global variables
let selectedFiles = [];

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeUpload();
    initializeAdvancedOptions();
});

/**
 * Initialize file upload functionality
 */
function initializeUpload() {
    const uploadBox = document.getElementById('uploadBox');
    const fileInput = document.getElementById('fileInput');
    
    if (!uploadBox || !fileInput) return;
    
    // File input change handler
    fileInput.addEventListener('change', function(e) {
        handleFileSelect(e.target.files);
    });
    
    // Drag and drop handlers
    uploadBox.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadBox.classList.add('drag-over');
    });
    
    uploadBox.addEventListener('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadBox.classList.remove('drag-over');
    });
    
    uploadBox.addEventListener('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadBox.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        handleFileSelect(files);
    });
}

/**
 * Initialize advanced options
 */
function initializeAdvancedOptions() {
    const epsSlider = document.getElementById('epsValue');
    const epsDisplay = document.getElementById('epsDisplay');
    
    if (epsSlider && epsDisplay) {
        epsSlider.addEventListener('input', function() {
            epsDisplay.textContent = this.value;
        });
    }
}

/**
 * Handle file selection
 */
function handleFileSelect(files) {
    selectedFiles = Array.from(files);
    displayFileList();
}

/**
 * Display selected files
 */
function displayFileList() {
    const fileList = document.getElementById('fileList');
    const fileListContent = document.getElementById('fileListContent');
    
    if (!fileList || !fileListContent) return;
    
    if (selectedFiles.length === 0) {
        fileList.style.display = 'none';
        return;
    }
    
    fileList.style.display = 'block';
    fileListContent.innerHTML = '';
    
    selectedFiles.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        
        const icon = getFileIcon(file.name);
        const size = formatFileSize(file.size);
        
        fileItem.innerHTML = `
            <span class="file-item-icon">${icon}</span>
            <span class="file-item-name">${file.name}</span>
            <span class="file-item-size">${size}</span>
        `;
        
        fileListContent.appendChild(fileItem);
    });
}

/**
 * Get icon for file type
 */
function getFileIcon(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    const imageExts = ['jpg', 'jpeg', 'png', 'gif', 'bmp'];
    const videoExts = ['mp4', 'avi', 'mov', 'mkv'];
    
    if (imageExts.includes(ext)) return '🖼️';
    if (videoExts.includes(ext)) return '🎬';
    return '📄';
}

/**
 * Format file size to human readable
 */
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

/**
 * Upload files and start processing
 */
async function uploadFiles() {
    if (selectedFiles.length === 0) {
        alert('Please select files first');
        return;
    }
    
    // Show processing section
    showProcessing('Uploading files...');
    
    // Prepare form data
    const formData = new FormData();
    selectedFiles.forEach(file => {
        formData.append('files', file);
    });
    
    try {
        // Upload files
        updateProgress(30, 'Uploading files...');
        const uploadResponse = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const uploadResult = await uploadResponse.json();
        
        if (!uploadResult.success) {
            throw new Error(uploadResult.message);
        }
        
        console.log('Upload successful:', uploadResult);
        
        // Start processing
        updateProgress(50, 'Processing faces...');
        await processFaces();
        
    } catch (error) {
        console.error('Error:', error);
        alert('Error: ' + error.message);
        hideProcessing();
    }
}

/**
 * Process faces using backend API
 */
async function processFaces() {
    try {
        // Get advanced options
        const eps = parseFloat(document.getElementById('epsValue').value);
        const minSamples = parseInt(document.getElementById('minSamples').value);
        const processVideos = document.getElementById('processVideos').checked;
        
        updateProgress(60, 'Detecting faces...');
        
        const response = await fetch('/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                eps: eps,
                min_samples: minSamples,
                process_videos: processVideos
            })
        });
        
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.message);
        }
        
        console.log('Processing complete:', result);
        
        // Show success
        updateProgress(100, 'Complete!');
        
        setTimeout(() => {
            window.location.href = '/results';
        }, 1000);
        
    } catch (error) {
        console.error('Processing error:', error);
        throw error;
    }
}

/**
 * Show processing section
 */
function showProcessing(message) {
    const processingSection = document.getElementById('processingSection');
    const uploadSection = document.querySelector('.upload-section');
    
    if (processingSection) {
        processingSection.style.display = 'block';
        updateProgress(10, message);
    }
    
    if (uploadSection) {
        uploadSection.style.display = 'none';
    }
}

/**
 * Hide processing section
 */
function hideProcessing() {
    const processingSection = document.getElementById('processingSection');
    const uploadSection = document.querySelector('.upload-section');
    
    if (processingSection) {
        processingSection.style.display = 'none';
    }
    
    if (uploadSection) {
        uploadSection.style.display = 'block';
    }
}

/**
 * Update progress bar and status
 */
function updateProgress(percent, message) {
    const progressFill = document.getElementById('progressFill');
    const processingStatus = document.getElementById('processingStatus');
    
    if (progressFill) {
        progressFill.style.width = percent + '%';
    }
    
    if (processingStatus && message) {
        processingStatus.textContent = message;
    }
}

/**
 * Toggle advanced options panel
 */
function toggleAdvanced() {
    const panel = document.getElementById('advancedPanel');
    if (panel) {
        if (panel.style.display === 'none' || panel.style.display === '') {
            panel.style.display = 'block';
        } else {
            panel.style.display = 'none';
        }
    }
}

/**
 * Check application status (for results page)
 */
async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const status = await response.json();
        console.log('App status:', status);
        return status;
    } catch (error) {
        console.error('Status check error:', error);
        return null;
    }
}
