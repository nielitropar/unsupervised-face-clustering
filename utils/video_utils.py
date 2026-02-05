"""
Video Processing Utilities
--------------------------
This module provides functions for:
- Extracting frames from video files
- Detecting faces in video frames
- Processing videos for face grouping
"""

import cv2
import os
import numpy as np
from typing import List, Tuple, Optional
import face_recognition


def is_valid_video(file_path: str) -> bool:
    """
    Check if a file is a valid video.
    
    Args:
        file_path (str): Path to file
        
    Returns:
        bool: True if valid video, False otherwise
    """
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    ext = os.path.splitext(file_path)[1].lower()
    return ext in valid_extensions


def get_videos_from_folder(folder_path: str) -> List[str]:
    """
    Get all valid video files from a folder.
    
    Args:
        folder_path (str): Path to folder
        
    Returns:
        List[str]: List of video file paths
    """
    video_files = []
    
    if not os.path.exists(folder_path):
        print(f"⚠️  Folder not found: {folder_path}")
        return video_files
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if is_valid_video(file_path):
                video_files.append(file_path)
    
    return sorted(video_files)


def extract_frames(video_path: str, output_folder: str, 
                   frame_interval: int = 30) -> List[str]:
    """
    Extract frames from a video at regular intervals.
    
    Args:
        video_path (str): Path to video file
        output_folder (str): Where to save extracted frames
        frame_interval (int): Extract every Nth frame (default: 30)
                             At 30fps, frame_interval=30 means 1 frame/second
        
    Returns:
        List[str]: Paths to extracted frame images
        
    Example:
        >>> frames = extract_frames("video.mp4", "output/frames", frame_interval=30)
        >>> print(f"Extracted {len(frames)} frames")
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open video file
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print(f"⚠️  Could not open video: {video_path}")
        return []
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"\n🎬 Processing video: {os.path.basename(video_path)}")
    print(f"   Duration: {duration:.1f}s, FPS: {fps:.1f}, Total frames: {total_frames}")
    print(f"   Extracting every {frame_interval} frames...")
    
    extracted_frames = []
    frame_count = 0
    saved_count = 0
    
    while True:
        # Read next frame
        success, frame = video.read()
        
        if not success:
            break
        
        # Save frame at intervals
        if frame_count % frame_interval == 0:
            frame_filename = f"frame_{frame_count:06d}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            extracted_frames.append(frame_path)
            saved_count += 1
        
        frame_count += 1
    
    # Release video
    video.release()
    
    print(f"✅ Extracted {saved_count} frames from video")
    return extracted_frames


def extract_frames_by_time(video_path: str, output_folder: str, 
                           time_interval: float = 1.0) -> List[str]:
    """
    Extract frames from a video at regular time intervals.
    
    Args:
        video_path (str): Path to video file
        output_folder (str): Where to save extracted frames
        time_interval (float): Extract frame every N seconds (default: 1.0)
        
    Returns:
        List[str]: Paths to extracted frame images
    """
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Open video
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print(f"⚠️  Could not open video: {video_path}")
        return []
    
    # Get FPS
    fps = video.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Default fallback
    
    # Calculate frame interval based on time
    frame_interval = int(fps * time_interval)
    
    # Use the frame-based extraction
    video.release()
    return extract_frames(video_path, output_folder, frame_interval)


def detect_faces_in_video(video_path: str, frame_interval: int = 30) -> Tuple[List[np.ndarray], List[str], List[int]]:
    """
    Detect all faces in a video by processing frames.
    
    Args:
        video_path (str): Path to video file
        frame_interval (int): Process every Nth frame
        
    Returns:
        Tuple containing:
            - all_encodings: List of face encodings
            - frame_paths: List of frame image paths
            - face_indices: List of face indices within each frame
    """
    # Create temporary folder for frames
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    temp_folder = os.path.join("uploads", f"temp_{video_name}_frames")
    
    # Extract frames
    frame_paths = extract_frames(video_path, temp_folder, frame_interval)
    
    if not frame_paths:
        return [], [], []
    
    # Detect faces in extracted frames
    all_encodings = []
    source_frames = []
    face_indices = []
    
    print(f"\n🔍 Detecting faces in {len(frame_paths)} frames...")
    
    for idx, frame_path in enumerate(frame_paths):
        if (idx + 1) % 10 == 0:
            print(f"   Processing frame {idx + 1}/{len(frame_paths)}")
        
        # Load frame and detect faces
        image = face_recognition.load_image_file(frame_path)
        face_locations = face_recognition.face_locations(image, model='hog')
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        # Store detected faces
        for face_idx, encoding in enumerate(face_encodings):
            all_encodings.append(encoding)
            source_frames.append(frame_path)
            face_indices.append(face_idx)
    
    print(f"✅ Found {len(all_encodings)} faces in video")
    
    return all_encodings, source_frames, face_indices


def get_video_info(video_path: str) -> dict:
    """
    Get information about a video file.
    
    Args:
        video_path (str): Path to video file
        
    Returns:
        dict: Video information (fps, duration, resolution, etc.)
    """
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        return {}
    
    info = {
        'fps': video.get(cv2.CAP_PROP_FPS),
        'total_frames': int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    
    if info['fps'] > 0:
        info['duration'] = info['total_frames'] / info['fps']
    else:
        info['duration'] = 0
    
    video.release()
    return info


def cleanup_temp_frames(video_path: str):
    """
    Remove temporary frame files created during video processing.
    
    Args:
        video_path (str): Original video path
    """
    import shutil
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    temp_folder = os.path.join("uploads", f"temp_{video_name}_frames")
    
    if os.path.exists(temp_folder):
        try:
            shutil.rmtree(temp_folder)
            print(f"🧹 Cleaned up temporary frames for {video_name}")
        except Exception as e:
            print(f"⚠️  Could not clean up temp folder: {str(e)}")
