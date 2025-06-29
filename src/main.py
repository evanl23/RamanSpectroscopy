from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2 as cv
import ramanspy
from ramanspy.utils import wavelength_to_wavenumber
from scipy.interpolate import interp1d
from scipy.stats import linregress
from scipy.signal import find_peaks

app = FastAPI()

# Global calibration values
calibration = {
    "slope": 1.0,
    "intercept": 0.0,
    "laser_wavelength": 532.0  # in nm
}

"""
    Helper methods for pixel to wavenumber conversion
"""
def pixel_to_wavelength(pixels: np.ndarray, slope: int, intercept: int) -> np.ndarray:
    return slope * pixels + intercept

@app.post("/process-image")
async def upload_img(img: UploadFile = File(...)):
    """
        Upload observed image for a substance.
    """
    contents = await img.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv.imdecode(np_arr, cv.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image format"})

    spectrum = np.mean(img, axis=0) # Collapse on X-axis to show intensity
    pixels = np.arange((img.shape)[1]) # Get number of pixels
    averaged = np.mean(spectrum, axis=1) # Average all RGB values

    interp_func = interp1d(pixels, averaged, kind='linear') # Create interpolate function
    library_range = np.linspace(200, 1800, 1601) # Range of wavenumbers in library
    fitted = np.linspace(pixels[0], pixels[-1], len(library_range)) # New array in shape of library wavenumbers
    fitted_intensity = interp_func(fitted) # Fit averaged pixels onto new array

    wavelengths = pixel_to_wavelength(library_range, calibration["slope"], calibration["intercept"])
    wavenumbers = wavelength_to_wavenumber(wavelengths, calibration["laser_wavelength"])

    # Initialize spectral container object
    spectral_data, spectral_axis = fitted_intensity, library_range # NOTE: change from library_range to wavenumbers once calibration is complete
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

    return {
        "index": preprocessed_axis.tolist(),
        "vector": preprocessed_data.tolist(),
        "shape": len(preprocessed_data)
    }

@app.post("/calibration")
async def calibrate(img: UploadFile, laser_wavelength: int):
    contents = await img.read()
    np_arr = np.frombuffer(contents, np.uint8)
    ethanol_observed = cv.imdecode(np_arr, cv.IMREAD_COLOR)
    if ethanol_observed is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image format"})
    
    spectrum = np.mean(ethanol_observed, axis=0) # Collapse on X-axis to show intensity
    averaged = np.mean(spectrum, axis=1) # Average all RGB values
    pixels = np.arange((ethanol_observed.shape)[1]) # Get number of pixels

    interp_func = interp1d(pixels, averaged, kind='linear') # Create interpolate function
    fitted = np.linspace(pixels[0], pixels[-1], 1601) # New array in shape of library wavenumbers
    fitted_intensity = interp_func(fitted) # Fit averaged pixels onto new array

    peaks = find_peaks(fitted_intensity, prominence=5, ) # Find peaks of observed spectrum
    pixel_peaks = np.array(peaks[0])

    known = None # TODO: Load graph of ethanol spectrum
    known_peaks = find_peaks(known, prominence=5) 
    wavelengths = np.array(known_peaks[0]) # Get peaks of known ethanol spectrum

    slope, intercept, _, _, _ = linregress(pixel_peaks, wavelengths)

    calibration["slope"] = slope
    calibration["intercept"] = intercept
    calibration["laser_wavelength"] = laser_wavelength