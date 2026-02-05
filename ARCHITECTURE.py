"""
PROJECT ARCHITECTURE & LOGIC EXPLANATION
=========================================
Automatic Face Grouping Application

This document explains the technical architecture, algorithms, and logic
behind the automatic face grouping system.
"""

# ============================================================================
# TABLE OF CONTENTS
# ============================================================================
"""
1. System Architecture Overview
2. Face Detection Pipeline
3. Clustering Algorithm (DBSCAN)
4. File Organization Logic
5. Flask Application Flow
6. Frontend-Backend Communication
7. Video Processing Pipeline
8. Error Handling & Edge Cases
9. Performance Optimization
10. Security Considerations
"""


# ============================================================================
# 1. SYSTEM ARCHITECTURE OVERVIEW
# ============================================================================
"""
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE (Browser)                 │
│  - File upload (drag & drop)                                │
│  - Progress tracking                                         │
│  - Results display                                           │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP/AJAX
┌─────────────────────▼───────────────────────────────────────┐
│                   FLASK WEB SERVER (app.py)                  │
│  Routes:                                                     │
│  - /          → Homepage                                     │
│  - /upload    → File upload handler                          │
│  - /process   → Face processing endpoint                     │
│  - /results   → Results display                              │
│  - /download  → ZIP download                                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
┌───────▼──────┐ ┌───▼──────┐ ┌───▼──────────┐
│ face_utils   │ │clustering│ │ video_utils  │
│              │ │          │ │              │
│ - Detection  │ │ - DBSCAN │ │ - Frame ext. │
│ - Embeddings │ │ - Groups │ │ - Face det.  │
│ - Batch proc.│ │ - Stats  │ │ - Cleanup    │
└──────────────┘ └──────────┘ └──────────────┘
        │             │             │
        └─────────────┼─────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              FILE SYSTEM (uploads/ & output/)                │
│  uploads/          →  Temporary storage                      │
│  output/Person_1/  →  Organized by cluster                   │
│  output/Person_2/  →  Organized by cluster                   │
│  output/Unknown/   →  Outliers                               │
└─────────────────────────────────────────────────────────────┘
"""


# ============================================================================
# 2. FACE DETECTION PIPELINE
# ============================================================================
"""
STEP-BY-STEP PROCESS:

Input: Image file path
Output: 128-dimensional face embeddings

1. IMAGE LOADING
   - Read image using face_recognition.load_image_file()
   - Converts to RGB format (required by dlib)
   - Handles: JPG, PNG, BMP, GIF

2. FACE DETECTION
   - Uses dlib's HOG (Histogram of Oriented Gradients) detector
   - Alternative: CNN detector (more accurate, slower)
   - Returns: Bounding boxes (top, right, bottom, left)
   
   Technical details:
   - HOG features capture edge patterns
   - Sliding window approach
   - Multi-scale detection

3. FACE ENCODING
   - For each detected face:
     a) Crop face region
     b) Align face (eye positions)
     c) Pass through pre-trained ResNet
     d) Extract 128-d embedding vector
   
   The embedding space has properties:
   - Similar faces → close together
   - Different faces → far apart
   - Euclidean distance measures similarity

4. QUALITY CHECKS
   - Skip blurry faces
   - Skip extreme angles
   - Skip partially visible faces
   
Example embedding:
   [0.123, -0.456, 0.789, ..., 0.234]  (128 values)
"""


# ============================================================================
# 3. CLUSTERING ALGORITHM (DBSCAN)
# ============================================================================
"""
WHY DBSCAN?
-----------
Traditional K-means requires knowing K (number of people) in advance.
We DON'T know how many unique people exist in the photos!

DBSCAN advantages:
- Automatically finds number of clusters
- Handles outliers (Unknown folder)
- Density-based (finds natural groupings)
- No need to specify K

ALGORITHM LOGIC:
----------------

Given: N face embeddings, parameters eps and min_samples

1. CORE POINT IDENTIFICATION
   For each face embedding:
   - Count neighbors within distance 'eps'
   - If neighbors >= min_samples → CORE POINT
   - Otherwise → BORDER POINT or NOISE

2. CLUSTER FORMATION
   - Start from a core point
   - Add all reachable core points to same cluster
   - Add border points connected to core points
   - Repeat until all core points are clustered

3. NOISE CLASSIFICATION
   - Points not reachable from any cluster → NOISE
   - In our case: Unknown folder

PARAMETERS:
-----------
eps (epsilon):
   - Maximum distance to be considered "neighbors"
   - Lower = stricter matching
   - Typical: 0.5 (moderate)
   - Face distance < 0.6 generally means same person

min_samples:
   - Minimum faces needed to form a cluster
   - Default: 2 (at least 2 photos of same person)
   - Higher values = stricter grouping

EXAMPLE:
--------
Faces: [F1, F2, F3, F4, F5]
Distances:
   F1-F2: 0.3  ✓ Same person
   F1-F3: 0.8  ✗ Different people
   F2-F3: 0.7  ✗ Different people
   F4-F5: 0.4  ✓ Same person

With eps=0.5, min_samples=2:
   Cluster 0: [F1, F2]  → Person_1
   Cluster 1: [F4, F5]  → Person_2
   Noise: [F3]          → Unknown
"""


# ============================================================================
# 4. FILE ORGANIZATION LOGIC
# ============================================================================
"""
CHALLENGE: Group photos with multiple people

Example scenario:
   Photo1.jpg: Contains Person A and Person B
   Photo2.jpg: Contains Person A only
   Photo3.jpg: Contains Person B only

Solution: Copy group photos to EACH person's folder

ALGORITHM:
----------
1. CREATE FOLDERS
   For each cluster_id in {0, 1, 2, ..., -1}:
      If cluster_id == -1:
         Create "Unknown" folder
      Else:
         Create "Person_{cluster_id + 1}" folder

2. MAP FACES TO SOURCES
   face_id → source_image_path
   Multiple faces can come from same image!

3. COPY FILES
   For each cluster:
      For each face in cluster:
         source = source_images[face_id]
         Copy source to cluster folder
         
   If same image copied multiple times:
      Add suffix: img_1.jpg, img_2.jpg, etc.

4. TRACK DUPLICATES
   Use set() to remember what we've copied
   Avoid duplicate copies to same folder

EXAMPLE OUTPUT:
---------------
output/
├── Person_1/           (Cluster 0: 3 faces)
│   ├── photo1.jpg      (contains Person_1 + Person_2)
│   ├── photo2.jpg      (contains Person_1 only)
│   └── photo5.jpg      (contains Person_1 + Person_3)
│
├── Person_2/           (Cluster 1: 2 faces)
│   ├── photo1.jpg      (same file! contains Person_1 + Person_2)
│   └── photo3.jpg      (contains Person_2 only)
│
└── Unknown/            (Noise: 1 face)
    └── photo4.jpg      (unclear face)
"""


# ============================================================================
# 5. FLASK APPLICATION FLOW
# ============================================================================
"""
USER REQUEST FLOW:
------------------

1. USER LANDS ON HOMEPAGE
   GET / → render_template('index.html')
   - Shows upload interface
   - Loads JavaScript for drag-and-drop

2. USER SELECTS FILES
   Frontend JS:
   - Validates file types
   - Shows file list
   - Displays preview

3. USER CLICKS "UPLOAD & START PROCESSING"
   
   Step 3.1: Upload Files
   POST /upload
   - Frontend sends FormData with files
   - Backend saves to uploads/ folder
   - Returns: {success: true, images_count: N, videos_count: M}
   
   Step 3.2: Process Faces
   POST /process
   Request body: {eps: 0.5, min_samples: 2, process_videos: true}
   
   Backend processing:
   a) Collect all uploaded images
   b) Extract face encodings (face_utils.py)
   c) Process videos if enabled (video_utils.py)
   d) Cluster faces (clustering.py)
   e) Organize files into folders
   f) Return results
   
   Response: {
      success: true,
      total_faces: 150,
      num_people: 8,
      num_unknown: 12,
      cluster_stats: [...]
   }

4. USER VIEWS RESULTS
   Frontend redirects to: GET /results
   - Backend lists output/ folder structure
   - Shows statistics per person
   - Displays download button

5. USER DOWNLOADS RESULTS
   GET /download
   - Backend creates ZIP file
   - Includes all organized folders
   - Returns: face_groups_TIMESTAMP.zip

ERROR HANDLING:
---------------
- 400: Bad request (no files, invalid params)
- 413: File too large (>500MB)
- 500: Server error (processing failed)
- All errors return JSON with error message
"""


# ============================================================================
# 6. FRONTEND-BACKEND COMMUNICATION
# ============================================================================
"""
AJAX WORKFLOW:
--------------

JavaScript (main.js) → Flask (app.py)

1. FILE UPLOAD
   JavaScript:
      const formData = new FormData();
      files.forEach(f => formData.append('files', f));
      fetch('/upload', {method: 'POST', body: formData})
   
   Flask:
      files = request.files.getlist('files')
      # Save files...
      return jsonify({success: true})

2. PROCESSING
   JavaScript:
      fetch('/process', {
         method: 'POST',
         headers: {'Content-Type': 'application/json'},
         body: JSON.stringify({eps: 0.5, min_samples: 2})
      })
   
   Flask:
      data = request.get_json()
      eps = data.get('eps', 0.5)
      # Process...
      return jsonify({success: true, num_people: N})

3. PROGRESS UPDATES
   JavaScript updates progress bar:
      updateProgress(30, 'Uploading...')
      updateProgress(60, 'Detecting faces...')
      updateProgress(100, 'Complete!')

SECURITY:
---------
- secure_filename() sanitizes uploads
- File type validation (ALLOWED_EXTENSIONS)
- Size limit (MAX_CONTENT_LENGTH)
- No execution of uploaded files
"""


# ============================================================================
# 7. VIDEO PROCESSING PIPELINE
# ============================================================================
"""
VIDEO → FRAMES → FACES → CLUSTERS

STEP 1: FRAME EXTRACTION
   Input: video.mp4 (30fps, 60 seconds = 1800 frames)
   Extract every 30th frame = 60 frames (1 per second)
   
   Why sample frames?
   - Processing every frame is redundant
   - Same person appears in consecutive frames
   - 1 frame/second captures sufficient variation

STEP 2: FACE DETECTION IN FRAMES
   For each extracted frame:
   - Run face detection
   - Extract embeddings
   - Store: (encoding, frame_path, face_index)

STEP 3: MERGE WITH IMAGES
   all_encodings = [image_encodings] + [video_encodings]
   Cluster together!

STEP 4: ORGANIZE RESULTS
   Option A: Flat structure
      output/Person_1/ contains frames from all videos
   
   Option B: Hierarchical (not implemented)
      output/Person_1/
         video1_frames/
         video2_frames/

STEP 5: CLEANUP
   Delete temporary frame files after processing
   Keeps only organized results

PERFORMANCE:
------------
- 1-minute video (30fps) = ~60 extracted frames
- ~30-60 seconds processing time
- Memory: ~200-500MB per video
"""


# ============================================================================
# 8. ERROR HANDLING & EDGE CASES
# ============================================================================
"""
EDGE CASES HANDLED:
-------------------

1. NO FACES DETECTED
   - Return error: "No faces detected in uploaded files"
   - User sees helpful message

2. ONLY 1 FACE TOTAL
   - DBSCAN with min_samples=2 marks as noise
   - Goes to Unknown folder
   - Solution: Set min_samples=1

3. ALL FACES DIFFERENT (no duplicates)
   - Each face becomes its own cluster
   - Many Person_1, Person_2, ... folders with 1 photo each
   - Expected behavior for non-family photos

4. IDENTICAL TWINS
   - May be grouped together (faces very similar)
   - User can adjust eps to be stricter
   - Or manually separate after download

5. POOR QUALITY IMAGES
   - Face detection may fail
   - Skip silently and continue processing
   - Log warning

6. VERY LARGE UPLOADS
   - MAX_CONTENT_LENGTH = 500MB limit
   - Returns 413 error if exceeded

7. CORRUPTED FILES
   - Try-catch in image loading
   - Skip corrupted file
   - Continue with rest

8. SAME PERSON, DIFFERENT ANGLES/LIGHTING
   - Embeddings designed to handle this
   - Should cluster together if eps appropriate

9. GROUP PHOTOS
   - Detect multiple faces per image
   - Copy image to each person's folder
   - Works correctly!

10. NO CLUSTERS FOUND
    - All faces marked as noise
    - Only Unknown folder created
    - Suggests: relax eps parameter
"""


# ============================================================================
# 9. PERFORMANCE OPTIMIZATION
# ============================================================================
"""
BOTTLENECKS & SOLUTIONS:
------------------------

1. FACE DETECTION (SLOWEST)
   Bottleneck: dlib face detection ~0.5s per image
   Solutions:
   - Use HOG instead of CNN (3-5x faster)
   - Batch processing
   - Resize large images before detection
   - Multi-threading (future enhancement)

2. CLUSTERING
   Bottleneck: O(N²) distance calculations for DBSCAN
   Solutions:
   - sklearn optimized implementation
   - n_jobs=-1 (use all CPU cores)
   - For >10,000 faces, consider approximate methods

3. FILE I/O
   Bottleneck: Copying large images multiple times
   Solutions:
   - Use shutil.copy2 (efficient)
   - Consider symlinks (not used for compatibility)
   - Compress output ZIP

4. MEMORY USAGE
   Current: Load all images in RAM
   Solutions:
   - Process in batches of 100 images
   - Delete uploaded files after extraction
   - Use generators instead of lists

CURRENT PERFORMANCE:
--------------------
- 100 images: ~2-3 minutes
- 500 images: ~10-15 minutes
- 1000 images: ~25-30 minutes

With optimizations:
- GPU acceleration: 3-5x faster
- Batch processing: 2x faster
- Parallel processing: 2-4x faster (multi-core)
"""


# ============================================================================
# 10. SECURITY CONSIDERATIONS
# ============================================================================
"""
SECURITY MEASURES:
------------------

1. FILE UPLOAD SECURITY
   ✓ secure_filename() prevents path traversal
   ✓ Extension whitelist (no .exe, .sh, etc.)
   ✓ Size limit (prevent DoS)
   ✗ No virus scanning (add in production)

2. NO CODE EXECUTION
   ✓ Only read image data
   ✓ No eval() or exec()
   ✓ No shell commands on user input

3. TEMPORARY FILE HANDLING
   ✓ Files cleared on new upload
   ✓ No permanent storage
   ⚠ No encryption at rest (add for sensitive data)

4. NETWORK SECURITY
   ⚠ HTTP only (use HTTPS in production)
   ⚠ No authentication (add for multi-user)
   ⚠ No rate limiting (add to prevent abuse)

5. PRIVACY
   ✓ All processing is local
   ✓ No external API calls
   ✓ No data persistence (except temp)
   ⚠ Add option to auto-delete after download

PRODUCTION RECOMMENDATIONS:
---------------------------
- Deploy behind HTTPS (nginx + Let's Encrypt)
- Add user authentication (Flask-Login)
- Implement rate limiting (Flask-Limiter)
- Add virus scanning (ClamAV)
- Encrypt temporary storage
- Add audit logging
- Implement CSRF protection
- Add session timeouts
"""


# ============================================================================
# ALGORITHM COMPLEXITY ANALYSIS
# ============================================================================
"""
TIME COMPLEXITY:
----------------
N = number of images
F = total faces detected
C = number of clusters

Face Detection:     O(N)      Linear in images
Face Encoding:      O(F)      Linear in faces
DBSCAN Clustering:  O(F²)     Quadratic in faces (worst case)
File Organization:  O(F)      Linear in faces

Total: O(N + F²)

SPACE COMPLEXITY:
-----------------
Face Encodings:     O(F × 128)    128-d vectors
Cluster Labels:     O(F)          One label per face
File Copies:        O(F × avg_img_size)

Total: O(F × img_size)

For typical use:
- 1000 images, 2000 faces, avg 2MB/image
- Time: ~30 minutes
- Space: ~4GB RAM
"""


# ============================================================================
# FUTURE ENHANCEMENTS
# ============================================================================
"""
PLANNED IMPROVEMENTS:
---------------------

1. GPU Acceleration
   - Use CUDA for CNN face detection
   - 5-10x speedup possible

2. Real-time Processing
   - Process images as they upload
   - WebSocket for live progress

3. Face Quality Scoring
   - Detect blurry/unclear faces
   - Only cluster high-quality faces

4. Incremental Updates
   - Upload more photos later
   - Add to existing clusters

5. User Feedback Loop
   - Mark incorrect groupings
   - Retrain/adjust parameters

6. Export Formats
   - JSON metadata
   - CSV face database
   - Integration with photo management tools

7. Advanced Features
   - Age/gender detection
   - Emotion recognition
   - Face search by upload
"""


# ============================================================================
# CONCLUSION
# ============================================================================
"""
This application demonstrates:
- Computer vision (face detection)
- Machine learning (embeddings, clustering)
- Web development (Flask, AJAX)
- File handling (uploads, organization)
- UI/UX design (drag-and-drop, progress)

Key Innovation:
- ZERO manual labeling required
- Fully automatic clustering
- Handles complex scenarios (group photos, videos)

Perfect for:
- Family photo organization
- Event photography
- Security/surveillance
- Social media management
"""
