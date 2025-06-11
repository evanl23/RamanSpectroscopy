import ramanspy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_file = "observed.csv"

try:
    data = pd.read_csv(csv_file)

    # Parse and load data into spectral objects
    spectral_axis = data["Wavenumber"]
    spectral_data = data["Intensity"]

    # Initialize spectral container object
    raman_spectrum = ramanspy.SpectralContainer(spectral_data, spectral_axis)

    # Pipeline object for preprocessing
    pipeline = ramanspy.preprocessing.Pipeline([
        ramanspy.preprocessing.denoise.SavGol(window_length=7, polyorder=2),
        # ramanspy.preprocessing.baseline.ASLS(), NOTE: baseline correction??
    ])
    preprocessed = pipeline.apply(raman_spectrum) # Preprocess spectrum
    preprocessed_axis = preprocessed.spectral_axis # Get pre-processed spectral axis (wavenumbers)
    preprocessed_data = preprocessed.spectral_data # Get pre-processed spectral data

    # Plot preprocessed and processed spectrum data
    plt.plot(spectral_axis, spectral_data, label="Raw")
    plt.plot(preprocessed_axis, preprocessed_data, 'r', label="Pre-processed")
    plt.legend()
    plt.title("Observed spectrum raw vs pre-processed")
    plt.xlabel("Raman shift wavenumber (cm⁻¹)")
    plt.ylabel("Intensity (arb)")
    plt.show()

    data = np.column_stack((preprocessed_axis, preprocessed_data)) # Combine into 2D array
    header="Wavenumber,Intensity"
    np.savetxt("observed_preprocessed.csv", data, delimiter=",", header=header, comments='') # Save preprocessed data as csv file

except FileNotFoundError as e:
    print(f"Incorrect file path or CSV file does not exist: {e}")