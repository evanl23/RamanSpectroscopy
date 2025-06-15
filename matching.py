import numpy as np
from typing import List, Tuple
from scipy.signal import correlate
import pandas as pd
import matplotlib.pyplot as plt
import ast
from scipy.interpolate import interp1d

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

def match_library( observed: np.ndarray, 
                   library: List[Tuple[str, np.ndarray]], 
                   top_n: int = 5) -> List[Tuple[str, float, np.ndarray, np.ndarray]]:
    """
    Compare observed spectrum to a library and return top-N matches. 
    
    ASSUMES same length array, intensities normalized to 1, and Raman shift is standardized.

    Takes
        observed: np.ndarray - Preprocessed intensity values of observed spectrum
        library: List of (label, wavenumber spectrum) tuples
        top_n: Number of top matches to return

    Returns
        List of (label, similarity score, wavenumber, spectrum), sorted by score descending
    """
    match = []
    for label, ref_spectrum in library:
        score = cosine_similarity(observed, ref_spectrum)
        # Similarity threshold
        if score < 0.8:
            pass
        else:
            match.append((label, score, ref_spectrum))
    # Sort by descending similarity
    match.sort(key=lambda x: x[1], reverse=True)
    return match[:top_n]

# Read observed spectrum data from csv file
observed_data_path = "csv_files/observed_preprocessed.csv"
data = pd.read_csv(observed_data_path)
observed = (data["Intensity"]).to_numpy()
preprocessed_axis = (data["Wavenumber"]).to_numpy()

# Read wavenumber data from csv file
wavenumber_csv = "csv_files/wavenumbers.csv"
wavenumber = pd.read_csv(wavenumber_csv)
wavenumbers = (wavenumber["wavenumber"]).to_numpy() # Convert df into array

# Read library spectrum data from csv file
csv_file = "csv_files/library.psd"
library_csv = pd.read_csv(csv_file)
library_csv["intensity"] = library_csv["intensity"].apply(ast.literal_eval) # Convert string to array
library = list(library_csv.itertuples(index=False, name=None)) # Convert csv library to list of tuples List[(substance, intensity)]

# Interpolates observed intensity values to match the reference axis.
if len(wavenumbers) != len(observed):     
    # Ensure axis is sorted
    assert np.all(np.diff(preprocessed_axis) > 0), "Observed axis must be increasing"
    assert np.all(np.diff(wavenumbers) > 0), "Reference axis must be increasing"
    # Interpolate to match reference
    observed = np.interp(wavenumbers, preprocessed_axis, observed)

scores = match_library(observed, library, top_n=3)

for label, score, intensity in scores:
    print(f"{label}: similarity = {score:.4f}") # Print to 4 decimals
    plt.plot(wavenumbers, intensity, label=label, alpha=0.5)

plt.plot(wavenumbers, observed, 'g', label="Observed")
plt.title("Observed spectrum matched against library")
plt.ylabel("Intensity (arb)")
plt.xlabel("Raman shift wavenumber (cm⁻¹)")
plt.legend()
plt.show()