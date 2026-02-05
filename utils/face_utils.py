"""
Face Detection and Embedding Utilities
---------------------------------------
This module provides functions for:
- Detecting faces in images using face_recognition library
- Extracting 128-dimensional face embeddings
- Processing multiple images in batch
"""

import face_recognition
import cv2
import numpy as np
from typing import List, Tuple, Optional
import os


def detect_faces_in_image(image_path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Detect faces in a single image and return face encodings and locations.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: 
            - List of face encodings (128-d vectors)
            - List of face locations (top, right, bottom, left)
            
    Example:
        >>> encodings, locations = detect_faces_in_image("photo.jpg")
        >>> print(f"Found {len(encodings)} faces")
    """
    try:
        # Load image using face_recognition (RGB format)
        image = face_recognition.load_image_file(image_path)
        
        # Detect face locations using HOG (Histogram of Oriented Gradients)
        # model='hog' is faster, 'cnn' is more accurate but requires GPU
        face_locations = face_recognition.face_locations(image, model='hog')
        
        # Generate 128-dimensional face encodings for each detected face
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        return face_encodings, face_locations
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return [], []


def extract_face_chip(image_path: str, face_location: Tuple[int, int, int, int], 
                      output_path: str, margin: int = 20) -> bool:
    """
    Extract and save a face chip from an image.
    
    Args:
        image_path (str): Path to source image
        face_location (Tuple): Face location (top, right, bottom, left)
        output_path (str): Where to save the extracted face
        margin (int): Extra pixels to include around the face
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            return False
            
        top, right, bottom, left = face_location
        
        # Add margin around face
        height, width = image.shape[:2]
        top = max(0, top - margin)
        bottom = min(height, bottom + margin)
        left = max(0, left - margin)
        right = min(width, right + margin)
        
        # Extract face region
        face_chip = image[top:bottom, left:right]
        
        # Save extracted face
        cv2.imwrite(output_path, face_chip)
        return True
        
    except Exception as e:
        print(f"Error extracting face: {str(e)}")
        return False


def process_image_batch(image_paths: List[str]) -> Tuple[List[np.ndarray], List[str], List[int]]:
    """
    Process multiple images and extract all face encodings.
    
    Args:
        image_paths (List[str]): List of image file paths
        
    Returns:
        Tuple containing:
            - all_encodings: List of all face encodings found
            - source_images: List of source image paths (same length as encodings)
            - face_indices: List of face index within each image
            
    Example:
        >>> encodings, sources, indices = process_image_batch(image_list)
        >>> # encodings[0] is the 1st face from sources[0]
    """
    all_encodings = []
    source_images = []
    face_indices = []
    
    print(f"\n🔍 Processing {len(image_paths)} images...")
    
    for idx, image_path in enumerate(image_paths):
        # Show progress
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"   Processing image {idx + 1}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        # Detect faces in this image
        encodings, locations = detect_faces_in_image(image_path)
        
        # Store each detected face
        for face_idx, encoding in enumerate(encodings):
            all_encodings.append(encoding)
            source_images.append(image_path)
            face_indices.append(face_idx)
    
    print(f"✅ Total faces detected: {len(all_encodings)}")
    return all_encodings, source_images, face_indices


def is_valid_image(file_path: str) -> bool:
    """
    Check if a file is a valid image.
    
    Args:
        file_path (str): Path to file
        
    Returns:
        bool: True if valid image, False otherwise
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    ext = os.path.splitext(file_path)[1].lower()
    return ext in valid_extensions


def get_images_from_folder(folder_path: str) -> List[str]:
    """
    Get all valid image files from a folder.
    
    Args:
        folder_path (str): Path to folder
        
    Returns:
        List[str]: List of image file paths
    """
    image_files = []
    
    if not os.path.exists(folder_path):
        print(f"⚠️  Folder not found: {folder_path}")
        return image_files
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if is_valid_image(file_path):
                image_files.append(file_path)
    
    return sorted(image_files)


def calculate_face_similarity(encoding1: np.ndarray, encoding2: np.ndarray) -> float:
    """
    Calculate similarity between two face encodings.
    
    Args:
        encoding1 (np.ndarray): First face encoding
        encoding2 (np.ndarray): Second face encoding
        
    Returns:
        float: Distance between encodings (lower = more similar)
    """
    return np.linalg.norm(encoding1 - encoding2)
