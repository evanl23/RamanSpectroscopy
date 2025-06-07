import numpy as np
from typing import List, Tuple
from scipy.signal import correlate

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
        Compute cosine similarity between two 1D arrays. 
        Compares the angle between to vectors.
        Compares overall shape.
        Doesn't account for baseline shifts or offsets.
    """
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
        Compute the euclidean distance between two 1D arrays.
        Compares absolute distance (intensity).
        Penalizes small shifts or baseline offsets harshly. 
    """
    return np.linalg.norm(a - b)

def cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """
        Slides one spectrum over the other and computes similarity at each shift (lag) to find the best alignment.
        Finding the pattern between the vectors, even if peaks are shifted.
        Good for slightly shifted peaks and noisy/poorly-calibrated data.
        Computationally heavy.
    """
    correlation = correlate(a,b, mode='full')
    return np.max(correlation)

def match_library( observed: np.ndarray, library: List[Tuple[str, np.ndarray]], top_n: int = 5) -> List[Tuple[str, float]]:
    """
    Compare observed spectrum to a library and return top-N matches. 
    
    ASSUMES same length array, intensities normalized to 1, and Raman shift is standardized.

    Takes
        observed: np.ndarray - Preprocessed intensity values of observed spectrum
        library: List of (label, spectrum) tuples
        top_n: Number of top matches to return

    Returns
        List of (label, similarity score), sorted by score descending
    """
    match = []
    for label, ref_spectrum in library:
        score = cosine_similarity(observed, ref_spectrum )
        # Similarity threshold
        if score < 0.8:
            pass
        else:
            match.append((label, score))
    # Sort by descending similarity
    match.sort(key=lambda x: x[1], reverse=True)
    return match[:top_n]

observed = np.array([0.1, 0.4, 0.6, 0.2])
library = [
    ("Acetone", np.array([0.1, 0.4, 0.6, 0.2])),
    ("Ethanol", np.array([0.2, 0.3, 0.5, 0.3])),
    ("Water", np.array([0.0, 0.1, 0.1, 0.2]))
]

scores = match_library(observed, library, top_n=2)

for label, score in scores:
    print(f"{label}: similarity = {score:.4f}") # Print to 4 decimals
