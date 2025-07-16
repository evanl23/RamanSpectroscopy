# Raman Spectroscopy

Scripts intended to process an image captured by a handheld, iPhone-based Raman spectroscope with the intent of identifying unique chemical signatures. Here is a quick overview on [Raman spectroscopy](https://www.horiba.com/usa/scientific/technologies/raman-imaging-and-spectroscopy/raman-spectroscopy/).

## Theory
The handheld device will provide the excitation using a laser, and through lenses in the device, collect the emitted Raman scatter. This scatter is captured by an iPhone camera and is then preprocessed and vectorized. This vector is then used to query a library for the *N* most similar spectrum, thus shedding light to the checmical signatures of the observed substance. 

This is currently a work in progress, and thus is only a proof-of-concept. 

## Technology stack
Hardware is constructed using a 532 nanometer laser and several lenses following this scientific paper by [Dinesh Dhankhar](https://pubs.aip.org/aip/rsi/article/92/5/054101/1021535/Cell-phone-camera-Raman-spectrometer).

Preprocessing of the observed spectra is done using [RamanSPy](https://ramanspy.readthedocs.io/en/latest/overview.html). A Qdrant vector database is utilized for the querying, and an ECS Fargate service is used for the backend preprocessing and querying. 
