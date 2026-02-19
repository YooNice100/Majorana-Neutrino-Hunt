# Majorana Neutrino Hunt

## Overview
This repository contains the code for our DSC 180A Quarter 1 Capstone Project at UCSD. We are analyzing digitized high-purity germanium (HPGe) detector waveforms from the [Majorana Demonstrator Data Release](https://zenodo.org/records/8257027).

The primary goal is to identify and engineer new waveform parameters that can effectively separate **single-site events (SSE)** from **multi-site events (MSE)**. This project implements a modular pipeline to extract features in both the time and frequency domains.

---

## Project Structure
The project is organized into modular source code to ensure reproducibility:

* `data/`: Stores the HDF5 datasets (ignored by Git).
* `extracted_features_csv_files`: Contains the feature-engineered datasets and the core Jupyter Notebooks containing the training logic, hyperparameter tuning, and final performance evaluations for all models.
* `graphs/`: Output directory for feature histograms and plots.
* `src/`: Source code modules.
    * `experiments/`: Main execution scripts (e.g., `exp_all.py`).
    * `parameters/`: Feature extraction logic (Time & Frequency domain).
    * `utils/`: Helper functions for I/O, stats, and plotting.
* `Dockerfile`: Instructions for building the project container.
* README.md
* `requirements.txt`: List of Python dependencies and versions.

---

## Methodology & Features
We implement a pipeline to load raw waveforms, apply necessary masks (SSE vs. MSE), and compute the following features:

**Time Domain:**
* **Peak Width:** Width of the pulse between 25% and 75% height.
* **Energy Duration:** Window duration containing significant energy.
* **Drift Time:** Time from threshold trigger (tp0) to 50% max height.
* **AvsE:** Comparison of max current Amplitude vs. Energy.
* **Time to Peak:** Duration between the pulse start (tp0) and the waveform’s maximum amplitude.

**Frequency Domain:**
* **Peak Frequency:** The frequency with the highest magnitude.
* **Spectral Centroid (Weighted):** The center of mass of the amplitude spectrum.
* **HFER:** High-Frequency Energy Ratio.
* **Spectral Centroid (Power):** The center of mass of the power spectrum.
* **Band Power Ratio:** The ratio of high‑frequency power to low‑frequency power in a waveform.
* **Total Power:** Sum of all spectral power components.

**Tail Features:**
* **LQ80:** A shape parameter requiring pole-zero correction.
* **ND80:** Normalized difference at 80% rise time.
* **TFR:** How much the waveform’s tail is flattened by PZ correction.
* **TCD:** A ratio measuring charge stability between early and late waveform tail regions.
* **Tail Slope:** Linear decay rate of the waveform tail

**Gradient Features:**
* **Current Skewness:** Asymmetry of gradient waveform between tp0 and peak.
* **Current Kurtosis:** Peakedness of gradient waveform.
* **Current Width:** Temporal width of the gradient waveform near its maximum.
* **Peak Count:** Number of significant local maxima in gradient waveform.
* **Gradient Baseline Noise:** how noisy the baseline of gradient waveform.

---

## Getting Started

### 1. Data Setup
This project requires a large data file that is **not** included in the repository due to size constraints.

1.  Navigate to the Zenodo data release page: [https://zenodo.org/records/8257027](https://zenodo.org/records/8257027)
2.  Download the file named **`MJD_Train_2.hdf5`**.
3.  Place it inside the `data/` folder in the root directory:
    ```text
    /Majorana-Neutrino-Hunt
    ├── data/
    │   └── MJD_Train_2.hdf5
    ```

### 2. Running the Analysis
You can run this project using a local Python environment **OR** using Docker (recommended for reproducibility).

#### Option A: Docker (Recommended)
Ensure you have Docker Desktop installed.

1.  **Build the Image:**
    ```bash
    docker build -t majorana-hunt .
    ```

2.  **Run the Container:**
    Use the following command to run the analysis and save the resulting graphs to your local machine:
    ```bash
    docker run -v $(pwd)/graphs:/app/graphs majorana-hunt
    ```
    *Note: The `-v` flag mounts your local `graphs/` folder to the container, ensuring plots are saved to your machine before the container exits.*

#### Option B: Local Python Environment
Ensure you have Python 3.11 installed.

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Experiment:**
    Execute the module from the root directory:
    ```bash
    python -m src.experiments.exp_all
    ```

---

## Outputs
Upon execution, the script produces:

1.  **Console Stats:** T-statistics and P-values for each feature (comparing SSE vs. MSE distributions) are printed to the console to verify statistical significance.
2.  **Visualizations:** Histograms for every feature are automatically saved to the `graphs/` directory (e.g., `graphs/LQ80_hist.png`, `graphs/avse_hist.png`).

---

## Contributors
* Nomin Batrigal
* Prithvi Kochhar
* Jade Choi
* Eunice Cho
* Aobo Li
