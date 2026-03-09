# Majorana Neutrino Hunt

## Overview

This repository contains the code for our **DSC 180B Capstone Project at UC San Diego**.  
The goal of this project is to use machine learning to analyze waveform data from the **Majorana Demonstrator experiment** and identify characteristics of particle interactions in high-purity germanium detectors.

Our pipeline performs the following tasks:

1. Combine extracted waveform features into training datasets.
2. Train classification models to identify event labels.
3. Train regression models to predict event energy.
4. Apply trained models to the NPML dataset.
5. Generate energy spectrum visualizations.

---

## Installation & Environment Setup

To ensure complete reproducibility, this project is packaged using Docker. This guarantees that all dependencies (Python 3.11, pandas 2.2.3, scikit-learn 1.6.1, xgboost 3.1.2, lightgbm 4.6.0, etc.) are installed with their exact versions.

**1. Clone the repository:**
`git clone https://github.com/YooNice100/Majorana-Neutrino-Hunt.git`
`cd Majorana-Neutrino-Hunt`

**2. Build the Docker Image:**
Ensure you have Docker Desktop installed and running, then build the image:
`docker build -t majorana-pipeline .`

---

## Dataset

The raw waveform datasets are too large to store in the repository. Instead, this repository contains pre-extracted feature inputs located in:

`src/feature_inputs/`

These datasets are used by the automated pipeline to build the final training and testing datasets for the models.

---

## Project Structure

Majorana-Neutrino-Hunt/
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt
├── data/
│   └── .gitkeep
└── src/
    ├── __init__.py
    ├── data/
    │   └── build_combined_dataset.py
    ├── feature_inputs/
    │   ├── npml/
    │   ├── test/
    │   └── train/
    ├── graphs/
    │   └── .gitkeep
    ├── models/
    │   ├── run_classification.py
    │   ├── run_npml_pipeline.py
    │   └── run_regression.py
    ├── notebooks/
    │   ├── classification_dcr.ipynb
    │   ├── classification_high_avse.ipynb
    │   ├── classification_low_avse.ipynb
    │   ├── classification_lq.ipynb
    │   ├── regression_lightgbm.ipynb
    │   └── regression_xgboost.ipynb
    ├── parameters/
    │   ├── __init__.py
    │   ├── frequency_domain.py
    │   ├── gradient_features.py
    │   ├── tail_features.py
    │   ├── time_domain.py
    │   └── transforms.py
    ├── results/
    │   ├── .gitkeep
    │   ├── classification_metrics.csv
    │   └── combined_classification_predictions.csv
    └── visualization/
        ├── generate_npml_plots.py
        └── generate_plots.py

---

## Running the Experiments

Our Docker container is configured to automatically run the entire pipeline sequentially. You only need to run one command to execute the data processing, model training, NPML predictions, and plot generation.

**Execute the pipeline:**
`docker run majorana-pipeline`

*(Note: The `Dockerfile` handles the sequential execution of `build_combined_dataset.py`, `run_classification.py`, `run_regression.py`, `run_npml_pipeline.py`, and the visualization scripts).*

---

## Expected Outputs

After the Docker container finishes running the pipeline, it will produce the following outputs in their respective directories:

**Model Metrics & Predictions:**
* `src/results/classification_metrics.csv`
* `src/results/combined_classification_predictions.csv`

**Visualizations:**
* `src/graphs/energy_spectrum_all_events.png`
* `src/graphs/energy_spectrum_after_psd_cut.png`
* `src/graphs/npml_energy_spectrum.png`

---

## Contributors

* Eunice Cho
* Nomin Batrigal
* Prithvi Kochhar
* Jade Choi
* Aobo Li