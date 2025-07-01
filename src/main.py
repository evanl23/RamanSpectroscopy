# Load environment variables from .env
import os
from dotenv import load_dotenv
load_dotenv()

# Imports for ASGI
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

# Imports for array processing
import cv2 as cv
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import linregress
from scipy.signal import find_peaks

# Imports for raman preprocessing
import ramanspy
from ramanspy.utils import wavelength_to_wavenumber

# Imports for vector DB
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance, SearchRequest

# Set up aws
import json
import boto3
from botocore.exceptions import ClientError
s3 = boto3.resource('s3')

app = FastAPI()

qdrant_client = QdrantClient(
    url="https://cfb7b44e-4b8a-440d-ac31-9ddc410c618f.us-west-2-0.aws.cloud.qdrant.io:6333", 
    api_key=os.getenv("qdrant_api_key"),
)

# Global calibration values
calibration = {
    "slope": 1.0,
    "intercept": 0.0,
    "laser_wavelength": 532.0  # in nm
}

"""
    Helper methods
"""
def pixel_to_wavelength(pixels: np.ndarray, slope: int, intercept: int) -> np.ndarray:
    return slope * pixels + intercept

async def qdrant_query(query: np.ndarray, collection_name: str, top_k: int = 5) -> dict:
    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query.tolist(),
        limit=top_k  # Return k closest points
    )
    return results

def get_metadata(s3_key: str) -> dict:
    s3 = boto3.client('s3')
    try: 
        obj = s3.get_object(
            Bucket='raman-spectra-bucket',
            Key=s3_key
        )
        return json.loads(obj['Body'].read())
    except ClientError as e:
        print(e)
        return {}
    
def put_metadata(s3_key: str, data: dict) -> bool:
    s3 = boto3.client('s3')
    try:
        s3.put_object(
            Body=json.dumps(data),
            Bucket='raman-spectra-bucket',
            Key=s3_key,
            ContentType='application/json'
        )
    except ClientError as e:
        print(e)
        return False
    return True

def metadata_exists(s3_key: str) -> bool:
    s3 = boto3.client('s3')
    try:
        s3.head_object(Bucket='raman-spectra-bucket', Key=s3_key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        else:
            raise

@app.post("/process-image")
async def upload_img(img: UploadFile = File(...)):
    """
        Upload observed image for a substance.
    """
    contents = await img.read() # Load image
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

    matches = await qdrant_query(preprocessed_data, "raman_library", 5) # Query qdrant for 5 closest raman vectors
    points = matches.points
    hits = [{"id": p.id, "score": p.score, "payload": p.payload} for p in points]

    # Fetch meta data (chemical properties) from S3
    compounds = []
    for h in hits:
        meta = get_metadata(h["payload"]["s3_key"])
        compounds.append(meta)

    # Put observed image in S3 bucket

    return {
        "closest_compounds": compounds,
    }

@app.post("/calibration")
async def calibrate(img: UploadFile, laser_wavelength: int):
    contents = await img.read() # Load image
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

    return {"calibrated_slope": slope, "calibrated_intercept": intercept, "excitation_wavelength": laser_wavelength}
    
# Read library spectrum data from csv file
"""
import pandas as pd
import ast
csv_file = "csv_files/library.psd"
library_csv = pd.read_csv(csv_file)
library_csv["intensity"] = library_csv["intensity"].apply(ast.literal_eval) # Convert string to array
library = list(library_csv.itertuples(index=False, name=None)) # Convert csv library to list of tuples List[(substance, intensity)]
for idx, (label, ref_spectrum) in library:
    qdrant_client.upsert(
        collection_name="raman_library",
        points=[
            PointStruct(
                id=idx,
                vector=ref_spectrum,
                payload={"substance": label, "s3_key": f"{label}.json"},
            )
        ],
    )
"""

# Add meta data information for each unique compound
"""
import pandas as pd
csv_file = "csv_files/library.psd"
library_csv = pd.read_csv(csv_file)
unique_labels = library_csv["label"].unique()
print(len(unique_labels))
for label in unique_labels:
    s3_key = f"{label}.json"
    # Check if compound already exists in S3. Some compounds have more than 1 spectra. Only add one meta data for each unique name
    if metadata_exists(s3_key):
            continue
    # If not present, add to S3
    data = {
        "molecular_formula": None,
        "molar_mass": None,
        "boiling_point": None,
        "licensing": None,
        "full_resolution_spectra": None,
    }
    put_metadata(s3_key, data)
"""
