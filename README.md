# 📸 Automatic Face Grouping Application

A Flask-based web application that automatically detects, groups, and organizes photos by faces using unsupervised machine learning. **No training, no manual labeling, no person names required!**

---

## 🎯 Features

- **Automatic Face Detection**: Detects all faces in images and videos
- **Unsupervised Clustering**: Groups similar faces using DBSCAN algorithm
- **Smart Organization**: Creates person folders automatically
- **Multi-Face Handling**: If one photo has multiple people, copies it to each person's folder
- **Video Support**: Extracts frames from videos and detects faces
- **No Training Required**: Uses pre-trained face_recognition models
- **User-Friendly Interface**: Simple drag-and-drop web interface
- **Download Results**: Get organized photos as a ZIP file

---

## 📁 Project Structure

```
face-grouping-app/
│
├── app.py                  # Main Flask application
├── face_utils.py          # Face detection and encoding utilities
├── clustering.py          # DBSCAN clustering logic
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
├── templates/            # HTML templates
│   ├── index.html       # Upload page
│   └── results.html     # Results page
│
├── uploads/             # Temporary storage for uploaded files
├── output/              # Organized output folders
│   ├── Person_1/
│   ├── Person_2/
│   ├── Person_3/
│   └── Unknown/
│
└── organized_photos.zip # Downloaded ZIP file
```

---

## 🛠️ Installation & Setup

### Prerequisites

- **Python 3.8 - 3.11** (face_recognition has compatibility issues with Python 3.12+)
- **Visual Studio C++ Build Tools** (Windows) or **build-essential** (Linux)
- **CMake** (for dlib compilation)

### Step 1: Clone or Download the Project

```bash
# If you have the files, navigate to the project directory
cd face-grouping-app
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: Installing `dlib` and `face_recognition` can take 5-15 minutes as they compile from source.

#### Troubleshooting Installation

**Windows Issues:**
- If you get CMake errors, install Visual Studio Build Tools:
  - Download from: https://visualstudio.microsoft.com/downloads/
  - Select "Desktop development with C++"

**macOS Issues:**
```bash
brew install cmake
pip install --upgrade pip
pip install dlib
pip install face-recognition
```

**Linux Issues:**
```bash
sudo apt-get update
sudo apt-get install build-essential cmake
sudo apt-get install libopenblas-dev liblapack-dev
pip install dlib
pip install face-recognition
```

**Alternative: Pre-built Wheels**
```bash
# Try pre-built dlib for faster installation (Windows)
pip install dlib-binary
```

### Step 4: Run the Application

```bash
python app.py
```

You should see:
```
* Running on http://0.0.0.0:5000
* Debug mode: on
```

### Step 5: Open in Browser

Navigate to: **http://localhost:5000**

---

## 🚀 How to Use

### 1. Upload Files
- Click the upload area or drag & drop your photos/videos
- Supported formats: **JPG, PNG, JPEG, MP4, AVI, MOV**
- Click **"Upload Files"**

### 2. Process
- Click **"Start Processing"**
- The app will:
  - Detect all faces
  - Create face embeddings (128-dimensional vectors)
  - Cluster similar faces using DBSCAN
  - Organize photos into person folders

### 3. View Results
- See how many unique persons were detected
- View the distribution of photos per person
- Photos with multiple people are copied to each person's folder

### 4. Download
- Click **"Download Results"** to get a ZIP file
- Extract and use your organized photos!

---

## 🧠 How It Works

### Technical Workflow

```
1. UPLOAD
   ↓
2. FACE DETECTION (face_recognition library)
   - Uses HOG (Histogram of Oriented Gradients) algorithm
   - Detects face bounding boxes in each image/frame
   ↓
3. FACE ENCODING
   - Converts each face into a 128-dimensional vector
   - Uses pre-trained ResNet model
   ↓
4. CLUSTERING (DBSCAN)
   - Groups similar face encodings
   - Automatically determines number of persons
   - Parameters: eps=0.5, min_samples=2
   ↓
5. ORGANIZATION
   - Creates Person_1, Person_2, ... folders
   - Copies photos to appropriate folders
   - Handles multi-face photos
   ↓
6. RESULTS
   - Display statistics
   - Provide download link
```

### Key Algorithms

**1. Face Detection**
- Uses `face_recognition.face_locations()` with HOG model
- Alternative: CNN model (more accurate, slower, requires GPU)

**2. Face Encoding**
- Pre-trained dlib ResNet model
- Converts faces to 128-D embeddings
- Faces of the same person have similar embeddings

**3. DBSCAN Clustering**
- **eps (epsilon)**: Maximum distance between faces to group them (0.5)
- **min_samples**: Minimum faces needed to form a group (2)
- **Metric**: Euclidean distance between embeddings
- **Benefits**: 
  - No need to specify number of clusters
  - Automatically handles noise/outliers (Unknown folder)

---

## ⚙️ Configuration Options

### Adjust Clustering Sensitivity

In `clustering.py`, modify the `cluster_faces()` function:

```python
# Stricter grouping (fewer false matches, might split one person)
cluster_faces(face_data_list, eps=0.4, min_samples=3)

# More lenient grouping (might merge different people)
cluster_faces(face_data_list, eps=0.6, min_samples=1)

# Default (balanced)
cluster_faces(face_data_list, eps=0.5, min_samples=2)
```

### Video Frame Extraction Rate

In `face_utils.py`, change `frame_interval`:

```python
# Extract 1 frame every 30 frames (default)
process_videos(upload_folder, video_files, frame_interval=30)

# More frames (slower, more accurate)
process_videos(upload_folder, video_files, frame_interval=15)

# Fewer frames (faster, less accurate)
process_videos(upload_folder, video_files, frame_interval=60)
```

### Face Detection Model

In `face_utils.py`:

```python
# HOG model (CPU, faster, less accurate) - DEFAULT
face_locations = face_recognition.face_locations(image, model='hog')

# CNN model (GPU recommended, slower, more accurate)
face_locations = face_recognition.face_locations(image, model='cnn')
```

---

## 📊 Performance Tips

### For Large Datasets (1000+ photos)

1. **Use HOG model** (not CNN) for faster processing
2. **Increase video frame interval** to 60 or 90
3. **Process in batches**: Upload 100-200 photos at a time
4. **Use a GPU** if available (requires CUDA setup for CNN model)

### Typical Processing Times

| Dataset Size | Time (HOG, CPU) | Time (CNN, GPU) |
|--------------|----------------|----------------|
| 50 photos    | 30 sec - 1 min | 20-40 sec      |
| 200 photos   | 2-5 min        | 1-3 min        |
| 500 photos   | 10-15 min      | 5-8 min        |
| 1000 photos  | 20-30 min      | 10-15 min      |

*Times vary based on image resolution and number of faces per photo*

---

## 🐛 Troubleshooting

### "No faces detected"
- Ensure faces are visible and not too small
- Check image quality (not too blurry)
- Try photos with faces directly facing the camera

### "Too many Unknown faces"
- Decrease `eps` value for stricter matching
- Increase `min_samples` to require more instances

### "Same person split into multiple groups"
- Increase `eps` value for more lenient matching
- Ensure good quality photos with clear faces

### Memory Issues
- Process fewer files at a time
- Use lower resolution images
- Increase video frame interval

### Slow Processing
- Use HOG model instead of CNN
- Process videos separately from images
- Reduce image resolution before uploading

---

## 🎓 Understanding DBSCAN Parameters

### `eps` (epsilon)
- **What**: Maximum distance between face encodings to group them
- **Lower (0.3-0.4)**: Very strict, fewer false matches, might split same person
- **Medium (0.5)**: Balanced (recommended)
- **Higher (0.6-0.7)**: More lenient, might merge different people

### `min_samples`
- **What**: Minimum number of faces to form a group
- **1**: Every face can form a group (more groups, fewer Unknown)
- **2**: Need at least 2 similar faces (balanced)
- **3+**: Need multiple instances (strict, more Unknown)

### Interpreting Results

- **Person_1, Person_2, ...**: Confident groupings
- **Unknown**: Outliers that don't match any group confidently
  - Could be unique individuals appearing once
  - Could be poor quality faces
  - Could be extreme angles/lighting

---

## 🔒 Privacy & Security

- All processing happens **locally** on your machine
- No data is sent to external servers
- Uploaded files are stored temporarily and can be deleted
- No face data is permanently stored
- No cloud services or APIs are used

---

## 📝 Code Explanation

### `app.py`
- Flask routes and web interface logic
- Handles file uploads, processing triggers, and downloads
- Orchestrates the complete workflow

### `face_utils.py`
- Face detection using `face_recognition` library
- Face encoding (converting faces to 128-D vectors)
- Video frame extraction and processing
- Image loading and preprocessing

### `clustering.py`
- DBSCAN clustering implementation
- Automatic grouping of similar faces
- Cluster analysis and statistics
- Parameter tuning helpers

### `templates/index.html`
- Upload interface with drag-and-drop
- File validation
- Progress tracking
- Responsive design

### `templates/results.html`
- Display clustering results
- Statistics visualization
- Download functionality

---

## 🚀 Future Enhancements

Potential improvements:
- [ ] Add face annotation/bounding boxes visualization
- [ ] Allow manual correction of groupings
- [ ] Support for multiple clustering algorithms
- [ ] Real-time processing feedback
- [ ] Database integration for large datasets
- [ ] API endpoints for programmatic access
- [ ] Docker containerization
- [ ] Batch processing queue system
- [ ] Advanced filtering options

---

## 📚 Dependencies Explained

| Package | Purpose |
|---------|---------|
| `Flask` | Web framework for the application |
| `face-recognition` | Face detection and encoding (wraps dlib) |
| `dlib` | Machine learning library with face recognition models |
| `opencv-python` | Video processing and image manipulation |
| `scikit-learn` | DBSCAN clustering algorithm |
| `numpy` | Numerical operations on face encodings |
| `Pillow` | Image loading and processing |
| `cmake` | Required to compile dlib |

---

## 🤝 Contributing

This is a learning project. Feel free to:
- Fork and experiment
- Add new features
- Improve clustering algorithms
- Enhance the UI/UX
- Fix bugs

---

## 📜 License

This project is for educational purposes. 

Face recognition model credits:
- `dlib` - Davis King (Boost Software License)
- `face_recognition` - Adam Geitgey (MIT License)

---

## 🆘 Getting Help

If you encounter issues:
1. Check the Troubleshooting section above
2. Verify Python version (3.8-3.11)
3. Ensure all dependencies installed correctly
4. Check console logs for error messages

---

## 🎉 Credits

Built with:
- Python & Flask
- dlib face recognition
- scikit-learn DBSCAN
- OpenCV
- face_recognition library by Adam Geitgey

---

**Happy Organizing! 📸✨**
