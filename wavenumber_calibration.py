import numpy as np
import cv2 as cv
from scipy.stats import linregress
from scipy.signal import find_peaks

ethanol_observed = cv.imread("ethanol_observed.png")
assert ethanol_observed is not None, "File cannot be read, check path is correct and file exists."

spectrum = np.mean(ethanol_observed, axis=0) # Collapse on X-axis to show intensity
averaged = np.mean(spectrum, axis=1) # Average all RGB values
peaks = find_peaks(averaged, prominence=5, ) # Find peaks of observed spectrum
pixel_peaks = np.array(peaks[0])

known = None # TODO: Load graph of ethanol spectrum
known_peaks = find_peaks(known, prominence=5) 
wavelengths = np.array(known_peaks[0]) # Get peaks of known ethanol spectrum

slope, intercept, _, _, _ = linregress(pixel_peaks, wavelengths) 