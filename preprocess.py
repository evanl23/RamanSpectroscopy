import ramanspy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_file = "observed.csv"

try:
    data = pd.read_csv(csv_file)

    # Parse and load data into spectral objects
    spectral_axis = data["Wavenumber"]
    spectral_data = data["Intensity"]

    raman_spectrum = ramanspy.SpectralContainer(spectral_data, spectral_axis)
    pipeline = ramanspy.preprocessing.Pipeline([
        ramanspy.preprocessing.denoise.SavGol(window_length=7, polyorder=2),
        # ramanspy.preprocessing.baseline.ASLS(),
    ])
    preprocessed = pipeline.apply(raman_spectrum) # Pre-processed spectrum
    preprocessed_data = preprocessed.spectral_data # Get pre-processed spectral data

    # Plot preprocessed and processed spectrum data
    plt.plot(spectral_axis, spectral_data, label="Raw")
    plt.plot(spectral_axis, preprocessed_data, 'r', label="Pre-processed")
    plt.legend()
    plt.title("Observed spectrum raw vs pre-processed")
    plt.xlabel("Raman shift wavenumber (cm⁻¹)")
    plt.ylabel("Intensity (arb)")
    plt.show()

except FileNotFoundError as e:
    print(f"Incorrect file path or CSV file does not exist: {e}")