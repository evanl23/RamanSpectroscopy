import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("isopropanol.png") # Load image
assert img is not None, "File could not be read. Check if file exists"

spectrum = np.mean(img, axis=0) # Collapse on X-axis to show intensity
averaged = np.mean(spectrum, axis=1) # Average all RGB values

plt.plot(averaged) # Visualize Raman spectral
plt.title("Extracted Raman Spectrum")
plt.xlabel("Pixel (proxy for Raman shift)")
plt.ylabel("Intensity (RGB average)")
plt.show()

# TODO: convert x axis to Raman shift (cm^{-1}) and y axis to intensity

header = "Wavenumber,Intensity"
np.savetxt("observed.csv", averaged, delimiter=",", header=header, comments='') # Save as csv file