# unsupervised-face-clustering
auto-face-grouping



Act as a senior AI/ML + Python developer.

I want to build a Flask-based web application for AUTOMATIC FACE GROUPING
(CASE 2 – NO TRAINING, NO NAMES).

PROJECT REQUIREMENTS:

1. Overview
- The application should automatically analyze photos (and optionally videos)
  uploaded via a web interface.
- It must detect faces, group similar faces together, and rearrange media
  into different folders.
- No prior training data, no person names, and no manual labeling.

2. Core Functionality
- User uploads:
  a) A folder of images (jpg, png, jpeg)
  b) Optional: video files (mp4, avi)
- The system should:
  - Detect all faces in each image or video frame
  - Convert faces into embeddings using a pre-trained model
  - Perform unsupervised clustering (DBSCAN)
  - Automatically determine how many unique people exist
  - Create folders like:
      output/
        ├── Person_1/
        ├── Person_2/
        ├── Person_3/
        └── Unknown/
  - If a group photo contains multiple people, copy the same image
    into each detected person’s folder.
  - If a face does not confidently match any cluster, store it in "Unknown".

3. Technical Stack
- Backend: Python + Flask
- Face Detection & Embedding:
  - face_recognition (dlib-based, pre-trained)
- Clustering:
  - scikit-learn (DBSCAN)
- Image & Video Processing:
  - OpenCV
- File Handling:
  - OS / shutil

4. Flask Features
- Homepage with:
  - File/folder upload option
  - Start Processing button
- Backend routes:
  - /upload
  - /process
  - /results
- After processing:
  - Show list of generated person folders
  - Allow user to download the organized output folder

5. Video Handling (Optional but preferred)
- Just Create a folder with name video and move all the videos their
- Don't Detect and group faces from video frames
- Create a sub-folder per video:
    videos/
      ├── video1.mp4
      ├── video2.mkv

6. Constraints
- No deep learning training
- No manual person naming
- Must work by just uploading and running
- Optimized for family or small company photo collections

7. Deliverables
- Clear project folder structure
- Complete Flask app (app.py)
- Utility modules (face_utils.py, clustering.py, video_utils.py)
- HTML templates (index.html, result.html)
- Step-by-step explanation of logic
- Instructions to run in VS Code

8. Output Quality
- Beginner-friendly
- Well-commented code
- Production-style folder structure
- Error handling for images with no faces

Generate the full working Flask project accordingly.
