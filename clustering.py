"""
Face Clustering Module
Uses DBSCAN (Density-Based Spatial Clustering) to automatically group similar faces
No training or manual labeling required
"""

import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict


def cluster_faces(face_data_list, eps=0.5, min_samples=2):
    """
    Cluster faces using DBSCAN algorithm
    
    DBSCAN automatically determines the number of clusters and identifies outliers.
    It groups faces based on the distance between their encodings.
    
    Args:
        face_data_list: List of face data dictionaries with 'encoding' key
        eps: Maximum distance between two samples for them to be considered 
             in the same neighborhood. Lower values = stricter grouping.
             Typical range: 0.4-0.6 for face_recognition encodings
        min_samples: Minimum number of samples in a neighborhood for a point
                    to be considered a core point. Set to 1 or 2 for small datasets.
    
    Returns:
        Updated face_data_list with 'cluster_id' added to each face
        cluster_id = -1 means "Unknown" (outlier/noise)
        cluster_id >= 0 means assigned to a person group
    """
    
    if len(face_data_list) == 0:
        return []
    
    # Extract face encodings into a numpy array
    encodings = np.array([face_data['encoding'] for face_data in face_data_list])
    
    print(f"\nClustering {len(encodings)} face encodings...")
    print(f"Parameters: eps={eps}, min_samples={min_samples}")
    
    # Perform DBSCAN clustering
    # metric='euclidean' measures straight-line distance between face encodings
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    cluster_labels = clusterer.fit_predict(encodings)
    
    # Add cluster labels to face data
    for i, face_data in enumerate(face_data_list):
        face_data['cluster_id'] = int(cluster_labels[i])
    
    # Get statistics
    unique_clusters = set(cluster_labels)
    num_clusters = len(unique_clusters - {-1})  # Exclude noise (-1)
    num_noise = list(cluster_labels).count(-1)
    
    print(f"Results:")
    print(f"  - Found {num_clusters} unique person(s)")
    print(f"  - {num_noise} face(s) marked as Unknown (outliers)")
    
    # Show cluster distribution
    cluster_counts = defaultdict(int)
    for label in cluster_labels:
        if label != -1:
            cluster_counts[f"Person_{label + 1}"] += 1
        else:
            cluster_counts["Unknown"] += 1
    
    print(f"\nCluster distribution:")
    for person, count in sorted(cluster_counts.items()):
        print(f"  - {person}: {count} face(s)")
    
    return face_data_list


def fine_tune_clustering(face_data_list, strict_mode=False):
    """
    Alternative clustering with different parameters
    
    Args:
        face_data_list: List of face data
        strict_mode: If True, uses stricter parameters (fewer false positives)
    
    Returns:
        Clustered face data
    """
    if strict_mode:
        # Stricter: smaller eps, more samples needed
        # Better for avoiding false matches, but might split one person into multiple groups
        return cluster_faces(face_data_list, eps=0.4, min_samples=3)
    else:
        # More lenient: larger eps, fewer samples needed
        # Better for grouping all instances of a person, but might merge different people
        return cluster_faces(face_data_list, eps=0.6, min_samples=1)


def get_cluster_summary(clustered_data):
    """
    Generate a summary of clustering results
    
    Args:
        clustered_data: List of face data with cluster_id assigned
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_faces': len(clustered_data),
        'unique_persons': 0,
        'unknown_faces': 0,
        'clusters': defaultdict(list)
    }
    
    for face_data in clustered_data:
        cluster_id = face_data['cluster_id']
        
        if cluster_id == -1:
            summary['unknown_faces'] += 1
            summary['clusters']['Unknown'].append(face_data['source_file'])
        else:
            person_name = f"Person_{cluster_id + 1}"
            summary['clusters'][person_name].append(face_data['source_file'])
    
    summary['unique_persons'] = len(summary['clusters']) - (1 if 'Unknown' in summary['clusters'] else 0)
    
    return summary


def refine_clusters_by_source(clustered_data):
    """
    Post-processing: Ensure faces from the same source file in the same cluster
    are only counted once. This is useful for preventing duplicate copies.
    
    This function analyzes the data but doesn't modify file organization.
    The main app.py handles the actual file copying logic.
    
    Args:
        clustered_data: List of face data with cluster_id
    
    Returns:
        Dictionary mapping (source_file, cluster_id) -> list of face instances
    """
    refined = defaultdict(list)
    
    for face_data in clustered_data:
        key = (face_data['source_file'], face_data['cluster_id'])
        refined[key].append(face_data)
    
    return refined


def suggest_eps_value(face_data_list, sample_size=100):
    """
    Suggest an appropriate eps value based on face encoding distances
    
    This is a heuristic helper function. In practice, eps=0.5 works well
    for most face_recognition encodings, but this can help fine-tune.
    
    Args:
        face_data_list: List of face data
        sample_size: Number of random pairs to sample
    
    Returns:
        Suggested eps value
    """
    if len(face_data_list) < 2:
        return 0.5
    
    encodings = np.array([face_data['encoding'] for face_data in face_data_list])
    
    # Randomly sample pairs and compute distances
    from sklearn.metrics.pairwise import euclidean_distances
    
    distances = euclidean_distances(encodings)
    
    # Get upper triangle (avoid diagonal and duplicates)
    upper_triangle_indices = np.triu_indices_from(distances, k=1)
    all_distances = distances[upper_triangle_indices]
    
    # Use median distance as a baseline
    median_distance = np.median(all_distances)
    
    # Suggested eps is slightly below median to ensure tight clusters
    suggested_eps = median_distance * 0.8
    
    print(f"\nDistance analysis:")
    print(f"  - Median distance: {median_distance:.3f}")
    print(f"  - Suggested eps: {suggested_eps:.3f}")
    
    return round(suggested_eps, 2)
