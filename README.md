# 🎭 Automatic Face Grouping Application

A Flask-based web application that automatically groups faces from photos and videos without requiring any training data or manual labeling. Perfect for organizing family photos, event pictures, or any collection of images with people.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/flask-3.0-green)
![License](https://img.shields.io/badge/license-MIT-blue)

## 🌟 Features

- **No Training Required**: Just upload and go - no setup or configuration needed
- **Automatic Face Detection**: Detects all faces in images and video frames using dlib
- **Smart Clustering**: Groups similar faces using DBSCAN unsupervised learning
- **Video Support**: Extracts and processes faces from video files (MP4, AVI, MOV)
- **Group Photo Handling**: Automatically copies images with multiple people to each person's folder
- **Unknown Faces**: Separate folder for faces that couldn't be confidently matched
- **Web Interface**: Beautiful, responsive UI with drag-and-drop upload
- **Batch Processing**: Handles hundreds of images efficiently
- **Download Results**: Export organized folders as a ZIP file

## 🏗️ Project Structure

```
Image_classification_and_distribution_/
│
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── face_utils.py          # Face detection and embedding extraction
│   ├── clustering.py          # DBSCAN clustering logic
│   └── video_utils.py         # Video frame extraction and processing
│
├── templates/                  # HTML templates
│   ├── index.html             # Upload page
│   └── results.html           # Results display page
│
├── static/                     # Static files
│   ├── css/
│   │   └── style.css          # Application styles
│   └── js/
│       └── main.js            # Frontend JavaScript
│
├── uploads/                    # Temporary upload storage
└── output/                     # Organized output folders
    ├── Person_1/
    ├── Person_2/
    ├── Person_3/
    └── Unknown/
```

## 🔧 Technical Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | Python 3.8+ with Flask |
| **Face Detection** | dlib (HOG-based detector) |
| **Face Recognition** | face_recognition library (128-d embeddings) |
| **Clustering** | scikit-learn DBSCAN |
| **Image Processing** | OpenCV, Pillow |
| **Video Processing** | OpenCV |
| **Frontend** | HTML5, CSS3, JavaScript (Vanilla) |

## 📋 Prerequisites

Before installation, ensure you have:

1. **Python 3.8 or higher**
   ```bash
   python --version
   ```

2. **Visual C++ Build Tools** (Windows only)
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Required for compiling dlib

3. **CMake** (for building dlib)
   ```bash
   pip install cmake
   ```

## 🚀 Installation

### Step 1: Clone or Navigate to Project

```bash
cd "c:\Data_Science_Programme\Machine_learning_models\Image_classification_and _distribution_"
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

**Option A: Standard Installation**
```bash
# Install CMake first
pip install cmake

# Install dlib
pip install dlib

# Install remaining dependencies
pip install -r requirements.txt
```

**Option B: If dlib installation fails (Windows)**
```bash
# Try pre-built wheel
pip install dlib-binary

# Then install other dependencies
pip install -r requirements.txt
```

**Option C: Using Conda (Recommended for Windows)**
```bash
# Create conda environment
conda create -n face_grouping python=3.9

# Activate environment
conda activate face_grouping

# Install dlib via conda
conda install -c conda-forge dlib

# Install remaining packages
pip install -r requirements.txt
```

### Step 4: Verify Installation

```python
# Test imports
python -c "import face_recognition; print('✅ face_recognition installed')"
python -c "import cv2; print('✅ OpenCV installed')"
python -c "import sklearn; print('✅ scikit-learn installed')"
```

## ▶️ Running the Application

### Method 1: Using Python

```bash
python app.py
```

### Method 2: Using Flask CLI

```bash
# Set environment variable (optional)
# Windows:
set FLASK_APP=app.py
set FLASK_ENV=development

# macOS/Linux:
export FLASK_APP=app.py
export FLASK_ENV=development

# Run
flask run
```

### Access the Application

Once running, open your browser and navigate to:
```
http://127.0.0.1:5000
```

You should see:
```
🎭 AUTOMATIC FACE GROUPING APPLICATION
========================================
Starting Flask server...
Open your browser and navigate to: http://127.0.0.1:5000
========================================
```

## 📖 How to Use

### 1. Upload Files

- Click **"Select Files"** or drag and drop images/videos
- Supported formats:
  - **Images**: JPG, JPEG, PNG, BMP, GIF
  - **Videos**: MP4, AVI, MOV, MKV
- Maximum file size: 500MB total

### 2. Configure Settings (Optional)

Click **"Advanced Options"** to adjust:

- **Similarity Threshold (eps)**: 
  - `0.4-0.5`: Strict matching (fewer false positives)
  - `0.5-0.6`: Moderate matching (default)
  - `0.6+`: Loose matching (may group different people)

- **Minimum Faces per Person**: 
  - `1`: Every face becomes its own cluster
  - `2`: Need at least 2 similar faces (default)
  - `3+`: Stricter grouping

- **Process Videos**: Enable/disable video processing

### 3. Start Processing

- Click **"Upload & Start Processing"**
- Wait for completion (progress bar shows status)

### 4. View Results

After processing, you'll see:
- Number of unique people found
- Number of photos in each person's folder
- Unknown/ungrouped faces

### 5. Download Results

Click **"Download All (ZIP)"** to get organized folders

## 🧠 How It Works

### Algorithm Overview

```
1. FACE DETECTION
   ↓
   For each image/video frame:
   - Use dlib's HOG-based detector
   - Extract face locations
   - Generate 128-dimensional embeddings

2. CLUSTERING
   ↓
   DBSCAN Algorithm:
   - No need to specify number of clusters
   - Automatically groups similar faces
   - Identifies outliers (Unknown folder)
   
3. ORGANIZATION
   ↓
   - Create folder per person (Person_1, Person_2, etc.)
   - Copy images to appropriate folders
   - Handle group photos (copy to multiple folders)
   - Store unmatched faces in Unknown folder
```

### Why DBSCAN?

- **Automatic cluster count**: Doesn't require knowing how many people exist
- **Density-based**: Groups faces that are close together in embedding space
- **Handles noise**: Outliers become "Unknown" instead of forcing bad matches
- **No training**: Works with any dataset immediately

### Face Embeddings

- Uses pre-trained ResNet model (from face_recognition library)
- Each face → 128-dimensional vector
- Similar faces have similar vectors
- Distance metric: Euclidean distance

## 🎯 Use Cases

1. **Family Photo Organization**
   - Upload vacation photos
   - Get folders organized by family member

2. **Event Photography**
   - Wedding photographers: group photos by guest
   - Party organizers: identify attendees

3. **Security & Surveillance**
   - Group security footage by person
   - Track individuals across multiple cameras

4. **Social Media Management**
   - Organize photos before posting
   - Find all photos of a specific person

## 🔍 Example Output

After processing, your `output/` folder will look like:

```
output/
├── Person_1/              # 45 photos
│   ├── IMG_001.jpg
│   ├── IMG_023.jpg
│   └── ...
│
├── Person_2/              # 32 photos
│   ├── IMG_005.jpg
│   ├── IMG_019.jpg
│   └── ...
│
├── Person_3/              # 28 photos
│   └── ...
│
└── Unknown/               # 12 photos
    └── ...
```

## ⚙️ Configuration

### Adjustable Parameters

Edit `app.py` to change defaults:

```python
# Flask configuration
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # Max upload size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'

# Clustering parameters (in clustering.py)
clusterer = FaceClusterer(
    eps=0.5,              # Similarity threshold
    min_samples=2         # Minimum faces per cluster
)

# Video processing (in video_utils.py)
frame_interval = 30       # Extract every 30th frame (1 per second at 30fps)
```

## 🐛 Troubleshooting

### Issue: dlib installation fails

**Solution**:
```bash
# Windows: Install Visual C++ Build Tools
# Then try:
pip install cmake
pip install dlib

# Or use conda:
conda install -c conda-forge dlib
```

### Issue: No faces detected

**Possible causes**:
1. Images too small or low quality
2. Faces not clearly visible
3. Extreme angles or occlusions

**Solution**:
- Use higher quality images
- Ensure faces are visible and well-lit
- Try adjusting detection parameters

### Issue: Too many groups created

**Solution**:
- Increase `eps` value (e.g., 0.6)
- Increase `min_samples` to 3 or higher

### Issue: Different people grouped together

**Solution**:
- Decrease `eps` value (e.g., 0.4)
- Use higher quality images
- Ensure faces are clearly visible

### Issue: Out of memory

**Solution**:
```python
# In face_utils.py, process in smaller batches
# Or use smaller images
```

## 📊 Performance

- **Face Detection**: ~0.5-1 second per image
- **Clustering**: ~1-2 seconds for 100 faces
- **Video Processing**: ~10-15 seconds per minute of video (30fps)
- **Memory**: ~2-4GB RAM for 1000 images

**Optimization Tips**:
- Use `model='hog'` for faster detection (CPU)
- Use `model='cnn'` for better accuracy (requires GPU)
- Reduce video frame extraction rate for faster processing

## 🔒 Privacy & Security

- All processing happens **locally** on your machine
- No data is sent to external servers
- Files are stored temporarily and can be deleted
- No permanent storage of uploaded files

**To clear all data**:
```bash
# Delete uploads and outputs
rm -rf uploads/* output/*

# Or manually delete the folders
```

## 🛠️ Development

### Running in Debug Mode

```python
# app.py
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

### Adding Custom Features

**Example: Add face count threshold**

```python
# In app.py
@app.route('/process', methods=['POST'])
def process_faces():
    min_face_count = request.json.get('min_face_count', 5)
    
    # Filter out clusters with fewer faces
    filtered_clusters = {
        k: v for k, v in cluster_assignments.items()
        if len(v) >= min_face_count
    }
```

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **face_recognition** library by Adam Geitgey
- **dlib** library by Davis King
- **scikit-learn** for DBSCAN implementation
- **OpenCV** for image and video processing

## 📧 Support

For issues or questions:
1. Check the troubleshooting section
2. Review GitHub issues
3. Contact: your-email@example.com

## 🚀 Future Enhancements

- [ ] Real-time video stream processing
- [ ] GPU acceleration for faster processing
- [ ] Face quality scoring
- [ ] Duplicate image detection
- [ ] Integration with cloud storage
- [ ] Mobile app version
- [ ] Multi-language support
- [ ] Export to photo management software

## 📚 References

- [face_recognition Documentation](https://github.com/ageitgey/face_recognition)
- [dlib Documentation](http://dlib.net/)
- [DBSCAN Algorithm](https://en.wikipedia.org/wiki/DBSCAN)
- [Flask Documentation](https://flask.palletsprojects.com/)

---

**Built with ❤️ using Python, Flask, and AI**

*Last Updated: February 2026*
