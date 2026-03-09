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

## Installation

**1. Clone the repository:**
`git clone https://github.com/YooNice100/Majorana-Neutrino-Hunt.git`
`cd Majorana-Neutrino-Hunt`

**2. Install dependencies:**
`pip install -r requirements.txt`

### Environment
This project was developed with Python 3.11 and the following core libraries:
* `pandas==2.2.3`
* `numpy==2.2.0`
* `scikit-learn==1.6.1`
* `scipy==1.15.1`
* `xgboost==3.1.2`
* `lightgbm==4.6.0`
* `matplotlib==3.9.3`

---

## Dataset

The raw waveform datasets are too large to store in the repository. Instead, this repository contains pre-extracted feature inputs located in:

`src/feature_inputs/`

These datasets are used to build the final training and testing datasets for the models.

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

## Running the Pipeline

You can run this project either using Docker for full automation or locally step-by-step.

### Method 1: Using Docker (Recommended)
Our Docker container is configured to automatically run the entire pipeline from dataset creation to final plot generation.

**1. Build the image:**
`docker build -t majorana-pipeline .`

**2. Run the pipeline:**
`docker run majorana-pipeline`

### Method 2: Local Python Environment
If you prefer to run the scripts individually, execute them from the root directory in the following order:

**Step 1: Build the Combined Dataset**
`python src/data/build_combined_dataset.py`

**Step 2: Train Classification Models**
`python src/models/run_classification.py`

**Step 3: Train Regression Models**
`python src/models/run_regression.py`

**Step 4: Run NPML Prediction Pipeline**
`python src/models/run_npml_pipeline.py`

**Step 5: Generate Visualizations**
`python src/visualization/generate_plots.py`
`python src/visualization/generate_npml_plots.py`

---

## Expected Outputs

After running the full pipeline, you should see the following files generated in their respective directories. If these files already exist, running the pipeline again will overwrite them with newly generated results.

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