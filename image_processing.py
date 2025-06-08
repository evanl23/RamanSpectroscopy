import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("isopropanol.png") # Load image
assert img is not None, "File could not be read. Check if file exists"

spectrum = np.mean(img, axis=0) # Collapse on X-axis to show intensity
pixels = np.arange((img.shape)[1]) # Get number of pixels
averaged = np.mean(spectrum, axis=1) # Average all RGB values

slope = 1 # TODO: Calibrated from wavenumber_calibration
intercept = 1 # TODO: Calibrated from wavenumber_calibration
def pixel_to_wavelength(pixels: np.ndarray) -> np.ndarray:
    return slope * pixels + intercept

laser_wavelength = 532 # nm TODO
def wavelength_to_wavenumber(wavelengths: np.ndarray) -> np.ndarray:
    return (1/laser_wavelength - 1/wavelengths) * 1e7

wavelengths = pixel_to_wavelength(pixels)
wavenumbers = wavelength_to_wavenumber(wavelengths)

plt.plot(wavenumbers, averaged) # Visualize Raman spectral
plt.title("Observed Raman spectrum")
plt.xlabel("Raman Shift Wavenumber (cm^-1)")
plt.ylabel("Intensity (RGB average)")
plt.show()

data = np.column_stack((wavenumbers, averaged))
header = "Wavenumber,Intensity"
np.savetxt("observed.csv", data, delimiter=",", header=header, comments='') # Save processed image data as csv file