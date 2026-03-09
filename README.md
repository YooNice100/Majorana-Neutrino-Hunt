# Majorana Neutrino Hunt

## Overview

This repository contains the code for our DSC 180B Capstone Project at UC San Diego.  
The goal of this project is to use machine learning to analyze waveform data from the Majorana Demonstrator experiment and identify characteristics of particle interactions in high-purity germanium detectors.

## Pipeline Summary

The Docker container executes the following scripts sequentially:

1. `build_combined_dataset.py` – merges extracted feature datasets into final training and test datasets.
2. `run_classification.py` – trains classification models to predict event labels.
3. `run_regression.py` – trains regression models to predict event energy.
4. `run_npml_pipeline.py` – applies trained models to the NPML dataset.
5. `generate_plots.py` and `generate_npml_plots.py` – produce energy spectrum visualizations.

---

## Installation & Environment Setup

To ensure complete reproducibility, this project is packaged using Docker. This guarantees that all dependencies (Python 3.11, pandas 2.2.3, scikit-learn 1.6.1, xgboost 3.1.2, lightgbm 4.6.0, etc.) are installed with their exact versions.

**1. Clone the repository:**
```bash
git clone https://github.com/YooNice100/Majorana-Neutrino-Hunt.git
cd Majorana-Neutrino-Hunt
```

**2. Build the Docker Image:**
Ensure you have Docker Desktop installed and running, then build the image:
```bash
docker build -t majorana .
```
*(Note: Building the Docker image may take several minutes the first time as dependencies are installed).*

**3. Run the pipeline:**
```bash
docker run majorana
```
*(Note: Running the full pipeline may take several minutes depending on system performance).*

---

## Dataset

The raw waveform datasets are too large to store in the repository. Instead, this repository contains pre-extracted feature datasets located in:

`src/feature_inputs/`

These datasets are used by the automated pipeline to build the final training and testing datasets for the models.

---

## Project Structure

* `.dockerignore`: Docker ignore rules.
* `.gitignore`: Git ignore rules.
* `Dockerfile`: Instructions for building the project container.
* `README.md`: Project documentation.
* `requirements.txt`: List of Python dependencies and versions.
* `data/`: Directory for raw datasets (ignored by Git).
* `src/`: Core source code modules.
    * `__init__.py`: Source package initialization.
    * `data/`: Data processing scripts (`build_combined_dataset.py`).
    * `feature_inputs/`: Pre-extracted feature datasets divided into `npml/`, `test/`, and `train/` subdirectories.
    * `graphs/`: Output directory for generated plots.
    * `models/`: Execution scripts for models (`run_classification.py`, `run_npml_pipeline.py`, `run_regression.py`).
    * `notebooks/`: Development and exploratory Jupyter notebooks containing training logic and evaluations for classification and regression.
    * `parameters/`: Feature extraction logic (`frequency_domain.py`, `gradient_features.py`, `tail_features.py`, `time_domain.py`, `transforms.py`).
    * `results/`: Output directory for model metrics and prediction CSVs (`classification_metrics.csv`, `combined_classification_predictions.csv`).
    * `visualization/`: Plot generation scripts (`generate_npml_plots.py`, `generate_plots.py`).
---

## Running the Experiments

Our Docker container is configured to automatically run the entire pipeline sequentially. You only need to run one command to execute the data processing, model training, NPML predictions, and plot generation.

**Execute the pipeline:**
```bash
docker run majorana
```

*(Note: The `Dockerfile` handles the sequential execution of `build_combined_dataset.py`, `run_classification.py`, `run_regression.py`, `run_npml_pipeline.py`, and the visualization scripts).*

---

## Expected Outputs

After the Docker container finishes running the pipeline, it will produce the following outputs in their respective directories.  
If these files already exist, running the pipeline again will overwrite them with newly generated results.

**Model Metrics & Predictions:**
* `src/results/classification_metrics.csv`
* `src/results/combined_classification_predictions.csv`
* `src/results/npml_predictions.csv`
* `src/results/regression_metrics.csv`
* `src/results/regression_predictions.csv`

**Visualizations:**
* `src/graphs/energy_spectrum_all_events.png`
* `src/graphs/energy_spectrum_after_psd_cut.png`
* `src/graphs/npml_lgb_all.png`
* `src/graphs/npml_lgb_psd_cut.png`
* `src/graphs/npml_xgb_all.png`
* `src/graphs/npml_xgb_psd_cut.png`


---

## Contributors

* Eunice Cho
* Nomin Batrigal
* Prithvi Kochhar
* Jade Choi
* Aobo Li