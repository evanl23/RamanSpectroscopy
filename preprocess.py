import ramanspy
import pandas as pd

csv_file = "observed.csv"

try:
    data = pd.read_csv(csv_file)

    # parse and load data into spectral objects
    spectral_data = data["Wavenumber"]
    spectral_axis = data["Intensity"]

    raman_spectrum = ramanspy.Spectrum(spectral_data, spectral_axis)

    
except FileNotFoundError as e:
    print("Incorrect file path or CSV file does not exist")