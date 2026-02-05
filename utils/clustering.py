"""
Face Clustering Utilities
-------------------------
This module handles unsupervised clustering of face embeddings using DBSCAN.
DBSCAN automatically determines the number of clusters without prior knowledge.
"""

import numpy as np
from sklearn.cluster import DBSCAN
from typing import List, Dict, Tuple
from collections import defaultdict


class FaceClusterer:
    """
    Unsupervised face clustering using DBSCAN algorithm.
    
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise):
    - Automatically finds the number of clusters
    - Can identify outliers (noise points)
    - No need to specify number of clusters in advance
    
    Attributes:
        eps (float): Maximum distance between two samples for one to be 
                     considered in neighborhood. Lower = stricter grouping
        min_samples (int): Minimum number of samples in a neighborhood for 
                          a point to be considered as a core point
    """
    
    def __init__(self, eps: float = 0.5, min_samples: int = 2):
        """
        Initialize the face clusterer.
        
        Args:
            eps (float): DBSCAN epsilon parameter (default: 0.5)
                        - 0.4-0.5: Strict matching (fewer false positives)
                        - 0.5-0.6: Moderate matching (balanced)
                        - 0.6+: Loose matching (may group different people)
            min_samples (int): Minimum samples per cluster (default: 2)
                              - 1: Every face forms its own cluster
                              - 2: At least 2 similar faces needed
                              - 3+: Stricter clustering
        """
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan = None
        self.labels_ = None
        self.n_clusters_ = 0
        self.n_noise_ = 0
        
    def fit(self, face_encodings: List[np.ndarray]) -> 'FaceClusterer':
        """
        Fit DBSCAN clustering to face encodings.
        
        Args:
            face_encodings (List[np.ndarray]): List of 128-d face embeddings
            
        Returns:
            self: The fitted clusterer
        """
        if len(face_encodings) == 0:
            print("⚠️  No face encodings provided for clustering")
            return self
        
        # Convert list to numpy array
        X = np.array(face_encodings)
        
        print(f"\n🎯 Clustering {len(face_encodings)} faces...")
        print(f"   Parameters: eps={self.eps}, min_samples={self.min_samples}")
        
        # Fit DBSCAN
        # metric='euclidean' uses standard Euclidean distance
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, 
                             metric='euclidean', n_jobs=-1)
        self.labels_ = self.dbscan.fit_predict(X)
        
        # Count clusters (label -1 is noise/unknown)
        unique_labels = set(self.labels_)
        self.n_clusters_ = len(unique_labels) - (1 if -1 in unique_labels else 0)
        self.n_noise_ = list(self.labels_).count(-1)
        
        print(f"✅ Clustering complete:")
        print(f"   📊 Unique people found: {self.n_clusters_}")
        print(f"   ❓ Unknown/outlier faces: {self.n_noise_}")
        
        return self
    
    def get_cluster_assignments(self) -> Dict[int, List[int]]:
        """
        Get cluster assignments as a dictionary.
        
        Returns:
            Dict[int, List[int]]: Mapping from cluster_id to list of face indices
            
        Example:
            >>> assignments = clusterer.get_cluster_assignments()
            >>> # {0: [0, 5, 12], 1: [1, 8], -1: [3, 7]}
            >>> # Person_1 has faces at indices 0, 5, 12
            >>> # Person_2 has faces at indices 1, 8
            >>> # Unknown has faces at indices 3, 7
        """
        clusters = defaultdict(list)
        
        for idx, label in enumerate(self.labels_):
            clusters[label].append(idx)
        
        return dict(clusters)
    
    def get_cluster_stats(self) -> List[Tuple[int, int]]:
        """
        Get statistics about each cluster.
        
        Returns:
            List[Tuple[int, int]]: List of (cluster_id, num_faces) sorted by size
        """
        assignments = self.get_cluster_assignments()
        stats = [(cluster_id, len(indices)) 
                 for cluster_id, indices in assignments.items()]
        
        # Sort by number of faces (descending), but put -1 (unknown) last
        stats.sort(key=lambda x: (x[0] == -1, -x[1]))
        
        return stats
    
    def predict_cluster(self, face_encoding: np.ndarray, 
                       all_encodings: List[np.ndarray]) -> int:
        """
        Predict which cluster a new face belongs to.
        
        Args:
            face_encoding (np.ndarray): New face encoding to classify
            all_encodings (List[np.ndarray]): All training encodings
            
        Returns:
            int: Predicted cluster label (-1 if unknown)
        """
        if self.labels_ is None:
            return -1
        
        # Find closest face in training set
        min_distance = float('inf')
        closest_idx = -1
        
        for idx, encoding in enumerate(all_encodings):
            distance = np.linalg.norm(face_encoding - encoding)
            if distance < min_distance:
                min_distance = distance
                closest_idx = idx
        
        # If distance is within eps, assign same cluster
        if min_distance <= self.eps and closest_idx != -1:
            return self.labels_[closest_idx]
        else:
            return -1  # Unknown
    
    def optimize_parameters(self, face_encodings: List[np.ndarray], 
                           desired_min_clusters: int = 2,
                           desired_max_clusters: int = 50) -> Tuple[float, int]:
        """
        Automatically find optimal DBSCAN parameters.
        
        Args:
            face_encodings (List[np.ndarray]): Face encodings to cluster
            desired_min_clusters (int): Minimum expected number of people
            desired_max_clusters (int): Maximum expected number of people
            
        Returns:
            Tuple[float, int]: Optimal (eps, min_samples)
        """
        best_eps = self.eps
        best_min_samples = self.min_samples
        best_score = 0
        
        # Try different parameter combinations
        eps_range = np.arange(0.4, 0.7, 0.05)
        min_samples_range = [2, 3, 4]
        
        for eps in eps_range:
            for min_samples in min_samples_range:
                temp_clusterer = FaceClusterer(eps=eps, min_samples=min_samples)
                temp_clusterer.fit(face_encodings)
                
                n_clusters = temp_clusterer.n_clusters_
                
                # Score based on reasonable cluster count
                if desired_min_clusters <= n_clusters <= desired_max_clusters:
                    score = 100 - abs(n_clusters - (desired_min_clusters + desired_max_clusters) / 2)
                    if score > best_score:
                        best_score = score
                        best_eps = eps
                        best_min_samples = min_samples
        
        print(f"🔧 Optimal parameters: eps={best_eps}, min_samples={best_min_samples}")
        return best_eps, best_min_samples
