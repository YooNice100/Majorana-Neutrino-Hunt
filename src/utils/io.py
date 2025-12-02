import os
import h5py
import numpy as np


def load_hdf5(path):
    """
    Loads an HDF5 file and returns all datasets as numpy arrays.

    Parameters
    ----------
    path : str
        Path to the .hdf5 file.

    Returns
    -------
    data : dict
        Keys are dataset names, values are numpy arrays.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with h5py.File(path, "r") as f:
        data = {key: np.array(f[key]) for key in f.keys()}

    return data


def sample_random_event(data):
    """
    Given a loaded HDF5 dictionary, select a random waveform
    and return all labels for that event.

    Parameters
    ----------
    data : dict
        Dictionary returned by load_hdf5().

    Returns
    -------
    event : dict
        Contains:
            waveform, energy, psd_low_avse, psd_high_avse,
            psd_dcr, psd_lq, tp0, detector, run_number, id
    """
    N = data["id"].shape[0]
    idx = np.random.randint(0, N)

    return {
        "index": idx,
        "waveform": data["raw_waveform"][idx],
        "energy": data["energy_label"][idx],
        "psd_low_avse": data["psd_label_low_avse"][idx],
        "psd_high_avse": data["psd_label_high_avse"][idx],
        "psd_dcr": data["psd_label_dcr"][idx],
        "psd_lq": data["psd_label_lq"][idx],
        "tp0": data["tp0"][idx],
        "detector": data["detector"][idx],
        "run_number": data["run_number"][idx],
        "id": data["id"][idx],
    }
