# 🔬 Majorana Neutrino Hunt — Quarter 1 Checkpoint

## Overview
This repository contains the code for our DSC 180A Quarter 1 Capstone Project. We are analyzing digitized high-purity germanium (HPGe) detector waveforms from the [Majorana Demonstrator Data Release](https://zenodo.org/records/8257027).

The primary goal is to identify and engineer new waveform parameters that can effectively separate **single-site events (SSE)** from **multi-site events (MSE)**.

---

## Getting Started

### 1. Data
This project requires a large data file that is **not** included in the repository.

1.  Navigate to the Zenodo data release page: [https://zenodo.org/records/8257027](https://zenodo.org/records/8257027)
2.  Download the file named **`MJD_Train_2.hdf5`**.

> **Note:** This file is approximately 2–3 GB. It is already included in the `.gitignore` file to prevent accidental uploads to GitHub. For this checkpoint, only this single training file is required.

### 2. Dependencies
Ensure you have all the required Python libraries installed. You can install them using pip:

* `numpy`
* `pandas`
* `h5py` (for reading the data file)
* `scipy`
* `matplotlib`
* `scikit-learn`

---

## How to Run

1.  **Configure File Path:**
    Inside the `script.ipynb` notebook (or your configuration file), update the file path to point to the `MJD_Train_2.hdf5` file you downloaded.

    *Example:*
    ```python
    file_path = "path/to/your/data/MJD_Train_2.hdf5"
    ```

2.  **Run Analysis:**
    Once the file path is correct and all dependencies are installed, you can run all the cells in `script.ipynb` to perform the analysis.

---

## Contributors
* Nomin Batrigal
* Prithvi Kochhar
* Jade Choi
* Eunice Cho
* Aobo Li