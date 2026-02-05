"""
Flask Application for Automatic Face Grouping
No training required - uses unsupervised clustering
"""

import os
import shutil
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from werkzeug.utils import secure_filename
import zipfile
from face_utils import process_images, process_videos
from clustering import cluster_faces
import json

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def is_video(filename):
    """Check if file is a video"""
    video_extensions = {'mp4', 'avi', 'mov'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in video_extensions


def cleanup_folders():
    """Clean up upload and output folders before new processing"""
    for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)


@app.route('/')
def index():
    """Homepage with upload interface"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads"""
    try:
        # Clean up previous uploads
        cleanup_folders()
        
        # Check if files were uploaded
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        files = request.files.getlist('files[]')
        
        if len(files) == 0 or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        uploaded_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_files.append(filename)
        
        if len(uploaded_files) == 0:
            return jsonify({'error': 'No valid files uploaded'}), 400
        
        return jsonify({
            'success': True,
            'message': f'{len(uploaded_files)} file(s) uploaded successfully',
            'files': uploaded_files
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/process', methods=['POST'])
def process():
    """Process uploaded files - detect faces, cluster, organize"""
    try:
        upload_folder = app.config['UPLOAD_FOLDER']
        output_folder = app.config['OUTPUT_FOLDER']
        
        # Get all uploaded files
        all_files = os.listdir(upload_folder)
        
        if len(all_files) == 0:
            return jsonify({'error': 'No files to process'}), 400
        
        # Separate images and videos
        image_files = [f for f in all_files if not is_video(f)]
        video_files = [f for f in all_files if is_video(f)]
        
        all_face_data = []
        
        # Process images
        if image_files:
            print(f"Processing {len(image_files)} images...")
            image_face_data = process_images(upload_folder, image_files)
            all_face_data.extend(image_face_data)
        
       
    # Handle videos: Just move them to a 'videos' folder without processing
        if video_files:
            print(f"Moving {len(video_files)} videos to output folder...")
            videos_output_path = os.path.join(output_folder, 'videos')
            os.makedirs(videos_output_path, exist_ok=True)
            
            for video_file in video_files:
                src_path = os.path.join(upload_folder, video_file)
                dst_path = os.path.join(videos_output_path, video_file)
                shutil.copy2(src_path, dst_path))
        
        if len(all_face_data) == 0:
            return jsonify({
                'error': 'No faces detected in uploaded media',
                'suggestion': 'Please upload images/videos containing visible faces'
            }), 400
        
        # Perform clustering
        print(f"Clustering {len(all_face_data)} detected faces...")
        clustered_data = cluster_faces(all_face_data)
        
        # Organize files into person folders
        organize_files(clustered_data, output_folder)
        
        # Get statistics
        stats = get_statistics(output_folder)
        
        return jsonify({
            'success': True,
            'message': 'Processing completed successfully',
            'stats': stats
        })
    
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return jsonify({'error': str(e)}), 500


def organize_files(clustered_data, output_folder):
    """
    Organize images into person folders based on clustering results
    If one image has multiple people, copy it to each person's folder
    """
    # Group faces by source file
    file_to_clusters = {}
    
    for face_data in clustered_data:
        source_file = face_data['source_file']
        cluster_id = face_data['cluster_id']
        
        if source_file not in file_to_clusters:
            file_to_clusters[source_file] = set()
        
        file_to_clusters[source_file].add(cluster_id)
    
    # Create person folders and copy files
    for source_file, cluster_ids in file_to_clusters.items():
        source_path = source_file
        
        for cluster_id in cluster_ids:
            if cluster_id == -1:
                folder_name = "Unknown"
            else:
                folder_name = f"Person_{cluster_id + 1}"
            
            dest_folder = os.path.join(output_folder, folder_name)
            os.makedirs(dest_folder, exist_ok=True)
            
            # Copy file to person folder
            dest_path = os.path.join(dest_folder, os.path.basename(source_file))
            
            # If file already exists (from another face in same image), skip
            if not os.path.exists(dest_path):
                shutil.copy2(source_path, dest_path)


def get_statistics(output_folder):
    """Get statistics about processed results"""
    stats = {
        'total_persons': 0,
        'unknown_count': 0,
        'person_counts': {}
    }
    
    if not os.path.exists(output_folder):
        return stats
    
    folders = os.listdir(output_folder)
    
    for folder in folders:
        folder_path = os.path.join(output_folder, folder)
        if os.path.isdir(folder_path):
            file_count = len(os.listdir(folder_path))
            
            if folder == "Unknown":
                stats['unknown_count'] = file_count
            else:
                stats['total_persons'] += 1
                stats['person_counts'][folder] = file_count
    
    return stats


@app.route('/results')
def results():
    """Show results page with statistics"""
    stats = get_statistics(app.config['OUTPUT_FOLDER'])
    return render_template('results.html', stats=stats)


@app.route('/download')
def download():
    """Create and download ZIP of organized output"""
    try:
        output_folder = app.config['OUTPUT_FOLDER']
        zip_path = 'organized_photos.zip'
        
        # Create ZIP file
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_folder)
                    zipf.write(file_path, arcname)
        
        return send_file(zip_path, as_attachment=True, download_name='organized_photos.zip')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Run Flask app in debug mode
    app.run(debug=True, host='0.0.0.0', port=5000)
