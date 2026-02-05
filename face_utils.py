"""
Face Detection and Encoding Utilities
Uses face_recognition library (dlib-based) for detecting and encoding faces
"""

import face_recognition
import cv2
import os
import numpy as np


def process_images(upload_folder, image_files):
    """
    Process all images to detect and encode faces
    
    Args:
        upload_folder: Path to folder containing uploaded images
        image_files: List of image filenames
    
    Returns:
        List of dictionaries containing face data:
        [
            {
                'source_file': '/path/to/image.jpg',
                'encoding': numpy_array,
                'location': (top, right, bottom, left)
            },
            ...
        ]
    """
    all_face_data = []
    
    for idx, image_file in enumerate(image_files):
        print(f"Processing image {idx + 1}/{len(image_files)}: {image_file}")
        
        image_path = os.path.join(upload_folder, image_file)
        
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Detect face locations
            # model can be 'hog' (faster, CPU) or 'cnn' (more accurate, GPU)
            face_locations = face_recognition.face_locations(image, model='hog')
            
            if len(face_locations) == 0:
                print(f"  No faces found in {image_file}")
                continue
            
            # Get face encodings (128-dimensional vectors)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            print(f"  Found {len(face_locations)} face(s)")
            
            # Store face data
            for face_encoding, face_location in zip(face_encodings, face_locations):
                all_face_data.append({
                    'source_file': image_path,
                    'encoding': face_encoding,
                    'location': face_location
                })
        
        except Exception as e:
            print(f"  Error processing {image_file}: {str(e)}")
            continue
    
    print(f"\nTotal faces detected in images: {len(all_face_data)}")
    return all_face_data


def process_videos(upload_folder, video_files, frame_interval=30):
    """
    Process videos by extracting frames and detecting faces
    
    Args:
        upload_folder: Path to folder containing uploaded videos
        video_files: List of video filenames
        frame_interval: Extract one frame every N frames (default: 30)
    
    Returns:
        List of face data dictionaries (same format as process_images)
    """
    all_face_data = []
    
    for idx, video_file in enumerate(video_files):
        print(f"Processing video {idx + 1}/{len(video_files)}: {video_file}")
        
        video_path = os.path.join(upload_folder, video_file)
        
        try:
            # Open video
            video_capture = cv2.VideoCapture(video_path)
            
            if not video_capture.isOpened():
                print(f"  Error: Could not open video {video_file}")
                continue
            
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            
            print(f"  Total frames: {total_frames}, FPS: {fps}")
            
            frame_count = 0
            processed_frames = 0
            video_faces = []
            
            while True:
                ret, frame = video_capture.read()
                
                if not ret:
                    break
                
                # Process every Nth frame
                if frame_count % frame_interval == 0:
                    # Convert BGR (OpenCV) to RGB (face_recognition)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces
                    face_locations = face_recognition.face_locations(rgb_frame, model='hog')
                    
                    if len(face_locations) > 0:
                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                        
                        for face_encoding, face_location in zip(face_encodings, face_locations):
                            video_faces.append({
                                'source_file': video_path,
                                'encoding': face_encoding,
                                'location': face_location,
                                'frame_number': frame_count
                            })
                    
                    processed_frames += 1
                
                frame_count += 1
            
            video_capture.release()
            
            print(f"  Processed {processed_frames} frames, found {len(video_faces)} face(s)")
            
            # For videos, we might want to deduplicate faces from the same person
            # appearing in multiple frames. We'll do this in clustering.
            all_face_data.extend(video_faces)
        
        except Exception as e:
            print(f"  Error processing video {video_file}: {str(e)}")
            continue
    
    print(f"\nTotal faces detected in videos: {len(all_face_data)}")
    return all_face_data


def draw_faces_on_image(image_path, face_locations, output_path):
    """
    Utility function to draw rectangles around detected faces
    Useful for debugging
    
    Args:
        image_path: Path to source image
        face_locations: List of face location tuples (top, right, bottom, left)
        output_path: Path to save annotated image
    """
    image = cv2.imread(image_path)
    
    for (top, right, bottom, left) in face_locations:
        # Draw rectangle around face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    
    cv2.imwrite(output_path, image)


def compare_faces(known_encoding, unknown_encoding, tolerance=0.6):
    """
    Compare two face encodings
    
    Args:
        known_encoding: First face encoding
        unknown_encoding: Second face encoding
        tolerance: How much distance between faces to consider a match (lower = more strict)
    
    Returns:
        Boolean indicating if faces match
    """
    distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]
    return distance <= tolerance
