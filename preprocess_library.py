import numpy as np
import pandas as pd
import ramanspy
from typing import Tuple

# NOTE: length of spectrum must all be the same, and the same as observed

def preprocess(spectral_data: np.ndarray, 
               spectral_axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
        Preprocess library spectrum in same format as observed.
    """
    raman_spectrum = ramanspy.SpectralContainer(spectral_data, spectral_axis)
    pipeline = ramanspy.preprocessing.Pipeline([
        ramanspy.preprocessing.denoise.SavGol(window_length=7, polyorder=2),
        ramanspy.preprocessing.baseline.ASLS(), 
        ramanspy.preprocessing.normalise.MinMax()
    ])
    preprocessed = pipeline.apply(raman_spectrum) # Pre-processed spectrum
    preprocessed_axis = preprocessed.spectral_axis
    preprocessed_data = preprocessed.spectral_data # Get pre-processed spectral data
    return preprocessed_data, preprocessed_axis

data = pd.read_csv("csv_files/raman_spectra_api_compounds.csv") # Load raw data
column = data.pop("label")
data.insert(loc=0, column="label", value=column) # Move label column to column position 0

intensity_columns = data.columns.drop("label")
data["intensity"] = data[intensity_columns].values.tolist() # Convert all intensities to an array
data = data[["label", "intensity"]] # Only keep label and intensity array columns

wavenumbers = [float(col) for col in intensity_columns] # Convert wavenumbers into array
header = "wavenumber"
np.savetxt("csv_files/wavenumbers.csv", wavenumbers, header=header, delimiter=",", comments='') # Save wavenumbers as separate csv file

library = list(data.itertuples(index=False, name=None)) # Convert csv library to list of tuples List[(substance, intensity)]

new_intensities = []
for substance, intensity in library:
    preprocessed, _ = preprocess(intensity, wavenumbers) # Preprocess library
    new_intensities.append(preprocessed.tolist())
data["intensity"] = new_intensities # Update data

data.to_csv("csv_files/library.csv", index=False)