"""
QUICK START GUIDE
==================
Automatic Face Grouping Application

Follow these steps to get your application running in VS Code.
"""

# ============================================================================
# STEP 1: OPEN TERMINAL IN VS CODE
# ============================================================================
# - Press Ctrl + ` (backtick) to open integrated terminal
# - Or go to: Terminal → New Terminal


# ============================================================================
# STEP 2: NAVIGATE TO PROJECT FOLDER
# ============================================================================
# Run this command in terminal:
#cd "c:\Data_Science_Programme\Machine_learning_models\Image_classification_and _distribution_"


# ============================================================================
# STEP 3: CREATE VIRTUAL ENVIRONMENT (RECOMMENDED)
# ============================================================================
# Create virtual environment:
python -m venv venv

# Activate it:
# On Windows (PowerShell):
venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
venv\Scripts\activate.bat

# You should see (venv) prefix in your terminal


# ============================================================================
# STEP 4: INSTALL DEPENDENCIES
# ============================================================================

# IMPORTANT: Install in this specific order!

# 1. Install CMake
pip install cmake

# 2. Install dlib (this may take 5-10 minutes)
pip install dlib

# If dlib fails on Windows, try:
pip install dlib-binary

# 3. Install all other dependencies
pip install -r requirements.txt


# ============================================================================
# STEP 5: VERIFY INSTALLATION
# ============================================================================
# Test if everything is installed correctly:
python -c "import face_recognition; print('✅ face_recognition works!')"
python -c "import cv2; print('✅ OpenCV works!')"
python -c "import sklearn; print('✅ scikit-learn works!')"
python -c "import flask; print('✅ Flask works!')"


# ============================================================================
# STEP 6: RUN THE APPLICATION
# ============================================================================
python app.py

# You should see:
# 🎭 AUTOMATIC FACE GROUPING APPLICATION
# Starting Flask server...
# Open your browser and navigate to: http://127.0.0.1:5000


# ============================================================================
# STEP 7: OPEN IN BROWSER
# ============================================================================
# Open your web browser and go to:
# http://127.0.0.1:5000
# or
# http://localhost:5000


# ============================================================================
# STEP 8: USE THE APPLICATION
# ============================================================================
# 1. Click "Select Files" or drag and drop images/videos
# 2. (Optional) Click "Advanced Options" to adjust settings
# 3. Click "Upload & Start Processing"
# 4. Wait for processing to complete
# 5. View results and download organized folders


# ============================================================================
# TROUBLESHOOTING
# ============================================================================

# Problem: "dlib installation failed"
# Solution:
#   1. Install Visual C++ Build Tools from:
#      https://visualstudio.microsoft.com/visual-cpp-build-tools/
#   2. Then try: pip install dlib
#   3. Or use conda: conda install -c conda-forge dlib

# Problem: "Module not found"
# Solution:
#   - Make sure virtual environment is activated (you see (venv))
#   - Reinstall: pip install -r requirements.txt

# Problem: "Address already in use"
# Solution:
#   - Port 5000 is busy
#   - Edit app.py and change: app.run(port=5001)

# Problem: "No faces detected"
# Solution:
#   - Use clear, high-quality images
#   - Ensure faces are visible and well-lit
#   - Try different images


# ============================================================================
# FOLDER STRUCTURE
# ============================================================================
# Your project should look like this:
#
# Image_classification_and_distribution_/
# ├── app.py                 ← Main Flask app
# ├── requirements.txt       ← Dependencies
# ├── README.md              ← Full documentation
# ├── QUICK_START.py         ← This file
# ├── .gitignore
# │
# ├── utils/                 ← Utility modules
# │   ├── __init__.py
# │   ├── face_utils.py      ← Face detection
# │   ├── clustering.py      ← DBSCAN clustering
# │   └── video_utils.py     ← Video processing
# │
# ├── templates/             ← HTML templates
# │   ├── index.html         ← Upload page
# │   └── results.html       ← Results page
# │
# ├── static/                ← CSS and JavaScript
# │   ├── css/
# │   │   └── style.css
# │   └── js/
# │       └── main.js
# │
# ├── uploads/               ← Temporary uploads (auto-created)
# └── output/                ← Organized results (auto-created)


# ============================================================================
# TESTING THE APPLICATION
# ============================================================================
# To test with sample images:
# 1. Create a folder with some photos containing faces
# 2. Upload through the web interface
# 3. Check the output/ folder for organized results


# ============================================================================
# STOPPING THE APPLICATION
# ============================================================================
# To stop the Flask server:
# - Press Ctrl + C in the terminal
# - The server will shut down gracefully


# ============================================================================
# ADDITIONAL COMMANDS
# ============================================================================

# Clear uploads and outputs:
# Windows (PowerShell):
Remove-Item -Path "uploads\*" -Recurse -Force
Remove-Item -Path "output\*" -Recurse -Force

# Check installed packages:
pip list

# Update packages:
pip install --upgrade -r requirements.txt

# Deactivate virtual environment:
deactivate


# ============================================================================
# NEED HELP?
# ============================================================================
# - Read the full README.md for detailed documentation
# - Check the troubleshooting section
# - Review code comments in each file
# - All functions are well-documented with docstrings


# ============================================================================
# ADVANCED USAGE
# ============================================================================

# Running with custom settings:
python app.py --port 8080

# Running in production mode:
# Install gunicorn: pip install gunicorn
# Run: gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Enable debug logging:
# In app.py, set: app.debug = True


# ============================================================================
# PERFORMANCE TIPS
# ============================================================================
# - For faster processing, use smaller images
# - Reduce video frame extraction rate (edit video_utils.py)
# - Process images in batches
# - Use 'hog' model for CPU, 'cnn' for GPU (requires CUDA)


# ============================================================================
# ENJOY! 🎭
# ============================================================================
# Your automatic face grouping application is ready to use!
# Upload your photos and let AI do the organizing.
