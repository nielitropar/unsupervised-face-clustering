"""
Flask Application for Automatic Face Grouping
=============================================
This application automatically groups faces from uploaded images and videos
without requiring any training data or manual labeling.

Author: AI/ML Developer
Date: February 2026
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import os
import shutil
from werkzeug.utils import secure_filename
import zipfile
from datetime import datetime

# Import custom utilities
from utils.face_utils import (
    process_image_batch, 
    get_images_from_folder, 
    detect_faces_in_image
)
from utils.clustering import FaceClusterer
from utils.video_utils import (
    detect_faces_in_video,
    get_videos_from_folder,
    cleanup_temp_frames
)

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload
app.config['ALLOWED_IMAGE_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'bmp', 'gif'}
app.config['ALLOWED_VIDEO_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)


def allowed_file(filename, file_type='image'):
    """Check if file extension is allowed."""
    if file_type == 'image':
        allowed = app.config['ALLOWED_IMAGE_EXTENSIONS']
    elif file_type == 'video':
        allowed = app.config['ALLOWED_VIDEO_EXTENSIONS']
    else:
        allowed = app.config['ALLOWED_IMAGE_EXTENSIONS'] | app.config['ALLOWED_VIDEO_EXTENSIONS']
    
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed


def clear_folders():
    """Clear upload and output folders before new processing."""
    for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)


@app.route('/')
def index():
    """Homepage with upload interface."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    """
    Handle file uploads (images and videos).
    
    Accepts:
        - Multiple image files
        - Multiple video files
        - Folder upload (via file input with webkitdirectory)
    
    Returns:
        JSON response with upload status
    """
    try:
        # Clear previous uploads
        clear_folders()
        
        # Get uploaded files
        files = request.files.getlist('files')
        
        if not files or files[0].filename == '':
            return jsonify({
                'success': False,
                'message': 'No files selected'
            }), 400
        
        uploaded_images = []
        uploaded_videos = []
        skipped_files = []
        
        # Process each uploaded file
        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                
                # Check file type
                if allowed_file(filename, 'image'):
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    uploaded_images.append(filename)
                    
                elif allowed_file(filename, 'video'):
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    uploaded_videos.append(filename)
                    
                else:
                    skipped_files.append(filename)
        
        # Prepare response
        response = {
            'success': True,
            'message': 'Files uploaded successfully',
            'images_count': len(uploaded_images),
            'videos_count': len(uploaded_videos),
            'skipped_count': len(skipped_files),
            'uploaded_images': uploaded_images[:5],  # Show first 5
            'uploaded_videos': uploaded_videos[:5]
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Upload error: {str(e)}'
        }), 500


@app.route('/process', methods=['POST'])
def process_faces():
    """
    Main processing route - performs face detection, clustering, and organization.
    
    Process:
        1. Collect all uploaded images and videos
        2. Extract faces from images
        3. Extract frames and faces from videos
        4. Cluster all faces using DBSCAN
        5. Organize files into person-specific folders
        6. Handle group photos (copy to multiple folders)
    
    Returns:
        JSON response with processing results
    """
    try:
        # Get processing parameters from request
        data = request.get_json() or {}
        eps = float(data.get('eps', 0.5))
        min_samples = int(data.get('min_samples', 2))
        process_videos = data.get('process_videos', True)
        
        print("\n" + "="*60)
        print("🚀 STARTING AUTOMATIC FACE GROUPING")
        print("="*60)
        
        # Step 1: Collect all images
        print("\n📁 Step 1: Collecting files...")
        image_paths = get_images_from_folder(app.config['UPLOAD_FOLDER'])
        video_paths = get_videos_from_folder(app.config['UPLOAD_FOLDER']) if process_videos else []
        
        print(f"   Found {len(image_paths)} images")
        print(f"   Found {len(video_paths)} videos")
        
        if len(image_paths) == 0 and len(video_paths) == 0:
            return jsonify({
                'success': False,
                'message': 'No valid images or videos found in upload folder'
            }), 400
        
        # Step 2: Extract face encodings from images
        print("\n👤 Step 2: Extracting faces from images...")
        all_encodings, source_images, face_indices = process_image_batch(image_paths)
        
        # Step 3: Process videos if any
        if video_paths:
            print("\n🎬 Step 3: Processing videos...")
            for video_path in video_paths:
                video_encodings, frame_paths, video_face_indices = detect_faces_in_video(
                    video_path, frame_interval=30
                )
                all_encodings.extend(video_encodings)
                source_images.extend(frame_paths)
                face_indices.extend(video_face_indices)
        
        total_faces = len(all_encodings)
        print(f"\n📊 Total faces detected: {total_faces}")
        
        if total_faces == 0:
            return jsonify({
                'success': False,
                'message': 'No faces detected in uploaded files'
            }), 400
        
        # Step 4: Cluster faces using DBSCAN
        print("\n🎯 Step 4: Clustering faces...")
        clusterer = FaceClusterer(eps=eps, min_samples=min_samples)
        clusterer.fit(all_encodings)
        
        cluster_assignments = clusterer.get_cluster_assignments()
        cluster_stats = clusterer.get_cluster_stats()
        
        # Step 5: Organize files into folders
        print("\n📂 Step 5: Organizing files into folders...")
        organize_files(cluster_assignments, source_images, face_indices)
        
        # Prepare results
        results = {
            'success': True,
            'message': 'Face grouping completed successfully',
            'total_faces': total_faces,
            'num_people': clusterer.n_clusters_,
            'num_unknown': clusterer.n_noise_,
            'cluster_stats': [
                {
                    'person': f'Person_{cid+1}' if cid != -1 else 'Unknown',
                    'faces': count
                }
                for cid, count in cluster_stats
            ]
        }
        
        print("\n" + "="*60)
        print("✅ PROCESSING COMPLETE")
        print("="*60)
        print(f"   Total unique people: {clusterer.n_clusters_}")
        print(f"   Unknown faces: {clusterer.n_noise_}")
        print("="*60 + "\n")
        
        return jsonify(results), 200
        
    except Exception as e:
        import traceback
        print(f"\n❌ Error during processing: {str(e)}")
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'message': f'Processing error: {str(e)}'
        }), 500


def organize_files(cluster_assignments, source_images, face_indices):
    """
    Organize files into person-specific folders.
    
    Args:
        cluster_assignments (dict): Mapping from cluster_id to face indices
        source_images (list): List of source image paths for each face
        face_indices (list): Face index within each image
    """
    output_folder = app.config['OUTPUT_FOLDER']
    
    # Create folders for each person
    for cluster_id in cluster_assignments.keys():
        if cluster_id == -1:
            folder_name = "Unknown"
        else:
            folder_name = f"Person_{cluster_id + 1}"
        
        cluster_folder = os.path.join(output_folder, folder_name)
        os.makedirs(cluster_folder, exist_ok=True)
    
    # Copy images to appropriate folders
    for cluster_id, face_ids in cluster_assignments.items():
        if cluster_id == -1:
            folder_name = "Unknown"
        else:
            folder_name = f"Person_{cluster_id + 1}"
        
        cluster_folder = os.path.join(output_folder, folder_name)
        
        # Track which source images we've already copied
        copied_images = set()
        
        for face_id in face_ids:
            source_image = source_images[face_id]
            
            # Generate unique filename
            base_name = os.path.basename(source_image)
            dest_path = os.path.join(cluster_folder, base_name)
            
            # If multiple faces from same image, add suffix
            if dest_path in copied_images:
                name, ext = os.path.splitext(base_name)
                counter = 1
                while dest_path in copied_images:
                    dest_path = os.path.join(cluster_folder, f"{name}_{counter}{ext}")
                    counter += 1
            
            # Copy file
            try:
                shutil.copy2(source_image, dest_path)
                copied_images.add(dest_path)
            except Exception as e:
                print(f"⚠️  Error copying {source_image}: {str(e)}")


@app.route('/results')
def results():
    """Display results page with organized folders."""
    try:
        # Get list of person folders
        output_folder = app.config['OUTPUT_FOLDER']
        
        if not os.path.exists(output_folder):
            return redirect(url_for('index'))
        
        # Collect folder statistics
        folders = []
        for folder_name in os.listdir(output_folder):
            folder_path = os.path.join(output_folder, folder_name)
            if os.path.isdir(folder_path):
                file_count = len([f for f in os.listdir(folder_path) 
                                 if os.path.isfile(os.path.join(folder_path, f))])
                folders.append({
                    'name': folder_name,
                    'count': file_count
                })
        
        # Sort: Unknown last, others by count descending
        folders.sort(key=lambda x: (x['name'] == 'Unknown', -x['count']))
        
        return render_template('results.html', folders=folders)
        
    except Exception as e:
        return f"Error loading results: {str(e)}", 500


@app.route('/download')
def download_results():
    """Create and download a ZIP file of all organized folders."""
    try:
        output_folder = app.config['OUTPUT_FOLDER']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_filename = f'face_groups_{timestamp}.zip'
        zip_path = os.path.join('uploads', zip_filename)
        
        # Create ZIP file
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_folder)
                    zipf.write(file_path, arcname)
        
        return send_file(zip_path, as_attachment=True, download_name=zip_filename)
        
    except Exception as e:
        return f"Error creating download: {str(e)}", 500


@app.route('/api/status')
def get_status():
    """Get current application status."""
    output_folder = app.config['OUTPUT_FOLDER']
    upload_folder = app.config['UPLOAD_FOLDER']
    
    status = {
        'output_exists': os.path.exists(output_folder) and len(os.listdir(output_folder)) > 0,
        'upload_exists': os.path.exists(upload_folder) and len(os.listdir(upload_folder)) > 0,
        'ready': True
    }
    
    return jsonify(status)


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({
        'success': False,
        'message': 'File too large. Maximum size is 500MB.'
    }), 413


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    return jsonify({
        'success': False,
        'message': 'Internal server error occurred.'
    }), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎭 AUTOMATIC FACE GROUPING APPLICATION")
    print("="*60)
    print("Starting Flask server...")
    print("Open your browser and navigate to: http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
