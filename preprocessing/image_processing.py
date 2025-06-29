import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from wavenumber_calibration import slope, intercept

observed_img_path = "img/isopropanol.png"
img = cv.imread(observed_img_path) # Load image
assert img is not None, "File could not be read. Check if file exists"

spectrum = np.mean(img, axis=0) # Collapse on X-axis to show intensity
pixels = np.arange((img.shape)[1]) # Get number of pixels
averaged = np.mean(spectrum, axis=1) # Average all RGB values

interp_func = interp1d(pixels, averaged, kind='linear') # Create interpolate function
library_range = np.linspace(200, 1800, 1601) # Range of wavenumbers in library
fitted = np.linspace(pixels[0], pixels[-1], len(library_range)) # New array in shape of library wavenumbers
fitted_intensity = interp_func(fitted) # Fit averaged pixels onto new array

slope = 1 # NOTE: Calibrated from wavenumber_calibration
intercept = 0 # NOTE: Calibrated from wavenumber_calibration
def pixel_to_wavelength(pixels: np.ndarray) -> np.ndarray:
    return slope * pixels + intercept

laser_wavelength = 532 # nm TODO: measure actual laser wavelength
def wavelength_to_wavenumber(wavelengths: np.ndarray) -> np.ndarray:
    return (1/laser_wavelength - 1/wavelengths) * 1e7

wavelengths = pixel_to_wavelength(library_range)
wavenumbers = wavelength_to_wavenumber(wavelengths) # can just use ramanspy.utils.wavelength_to_wavenumber

plt.plot(wavenumbers, fitted_intensity) # Visualize Raman spectral
plt.title("Observed Raman spectrum")
plt.xlabel("Raman Shift Wavenumber (cm⁻¹)")
plt.ylabel("Intensity (RGB average)")
plt.show()

data = np.column_stack((wavenumbers, fitted_intensity)) # Match intensity with wavenumber
header = "Wavenumber,Intensity"
np.savetxt("csv_files/observed.csv", data, delimiter=",", header=header, comments='') # Save processed image data as csv file