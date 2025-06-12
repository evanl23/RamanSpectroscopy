import ramanspy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_file = "csv_files/observed.csv"

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
        ramanspy.preprocessing.baseline.ASLS(), #NOTE: baseline correction??
        ramanspy.preprocessing.normalise.MinMax()
    ])
    preprocessed = pipeline.apply(raman_spectrum) # Preprocess spectrum
    preprocessed_axis = preprocessed.spectral_axis # Get pre-processed spectral axis (wavenumbers)
    preprocessed_data = preprocessed.spectral_data # Get pre-processed spectral data

    # Plot raw and preprocessed spectrum data
    fig, (one,two) = plt.subplots(1,2, figsize=(10,5))

    one.plot(spectral_axis, spectral_data, 'b', label="Raw")
    one.set_title("Raw spectrum")
    one.set_xlabel("Raman shift wavenumber (cm⁻¹)")
    one.set_ylabel("Intensity (arb)")

    two.plot(preprocessed_axis, preprocessed_data, 'r', label="Pre-processed")
    two.set_title("Pre-processed spectrum")
    two.set_xlabel("Raman shift wavenumber (cm⁻¹)")
    two.set_ylabel("Intensity (arb)")

    plt.show()

    data = np.column_stack((preprocessed_axis, preprocessed_data)) # Combine into 2D array
    header="Wavenumber,Intensity"
    np.savetxt("csv_files/observed_preprocessed.csv", data, delimiter=",", header=header, comments='') # Save preprocessed data as csv file

except FileNotFoundError as e:
    print(f"Incorrect file path or CSV file does not exist: {e}")