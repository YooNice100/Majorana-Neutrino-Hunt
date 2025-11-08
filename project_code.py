import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# compute tdrift, tdrift50, and tdrift10 points
def calculate_drift_times(waveform, tp0, step=0.1):
    """
    Compute tdrift (99.9%), tdrift50 (50%), and tdrift10 (10%) points 
    from the rising portion of a waveform.

    Parameters
    ----------
    waveform : np.ndarray
        The full waveform signal.
    tp0 : int
        Estimated start index of the rising edge (tp0).
    step : float
        Interpolation step size (default = 0.1).

    Returns
    -------
    tdrift, tdrift50, tdrift10
    """
    # Find waveform peak and slice the rising portion
    peak_index = np.argmax(waveform)
    waveform_rise = waveform[tp0:peak_index + 1]
    timeindex_rise = np.arange(0, len(waveform_rise))

    # Interpolate rising portion
    interp_func = interp1d(timeindex_rise, waveform_rise, kind='linear')
    new_timeindex_rise = np.arange(0, len(timeindex_rise) - 1, step)
    new_waveform_rise = interp_func(new_timeindex_rise)

    # Compute rise thresholds
    rise_peak = np.max(new_waveform_rise)
    tdrift = np.where(new_waveform_rise >= 0.999 * rise_peak)[0][0]
    tdrift50 = np.where(new_waveform_rise >= 0.5 * rise_peak)[0][0]
    tdrift10 = np.where(new_waveform_rise >= 0.1 * rise_peak)[0][0]

    return tdrift, tdrift50, tdrift10

# Double exponential decay model for fitting
def exponential(t, a, tau1, b, tau2):
    """
    Double exponential decay model.

    Returns:
        np.array: The evaluated double exponential decay at time t.
    """
    return a * np.exp(-t/tau1) + b * np.exp(-t/tau2)

# Applies pole-zero correction to a waveform.
def pole_zero_correction(waveform):
    """
    Args:
        raw waveform (np.array)
    Returns:
        waveform_pz (np.array): The PZ-corrected full waveform.
        waveform_tail_corrected (np.array): The PZ-corrected tail portion of the waveform.
    """
    # Identify the peak value
    peak_value = np.max(waveform)
    
    # Isolate the tail (starting at 98% of the peak)
    t98 = np.where(waveform >= 0.98 * peak_value)[0][0]
    
    # Generate the time index necessary for the fit (starting at 0 for the fit function)
    time_index = np.arange(0, len(waveform))
    tail_time = np.arange(0, time_index[-1] - t98 + 1)
    tail_values = waveform[t98:]

    # Fit the decay model to the raw tail values
    params, params_cov = curve_fit(exponential, tail_time, tail_values)

    # Calculate the correction factor and apply it
    f_decay = exponential(tail_time, *params)
    
    # Estimate the initial value of the tail (f_t0) from the first few samples near t98
    f_t0 = np.mean(waveform[t98:t98+5])
    
    # Calculate the inverse correction factor (f_pz). 
    # This factor, when multiplied by the tail, flattens the exponential decay.
    f_pz = f_t0 / f_decay
    
    # Apply the correction
    waveform_tail_corrected = tail_values * f_pz
    
    # Create the final corrected waveform
    waveform_pz = np.copy(waveform)
    waveform_pz[t98:] = waveform_tail_corrected
    
    return waveform_pz, waveform_tail_corrected

# calculate LQ80
def calculate_lq80(waveform, waveform_tail_corrected):
    """
    Calculate the LQ80 of a waveform given the pole-zero corrected tail.
    
    Parameters:
        waveform (np.ndarray): Original waveform
        waveform_tail_corrected (np.ndarray): Pole-zero corrected tail of waveform

    Returns:
        float: LQ80 value
    """
    tail_average = np.mean(waveform_tail_corrected)
    peak_value = np.max(waveform)
    t80 = np.where(waveform >= 0.80 * peak_value)[0][0]
    tpeak = np.where(waveform >= peak_value)[0][0]

    lq80_region = waveform[t80:tpeak+1]
    tail_mean_threshold = np.ones(len(lq80_region)) * tail_average

    waveform_underlq80 = np.trapz(lq80_region)
    tail_mean_area = np.trapz(tail_mean_threshold)

    lq80_region = tail_mean_area - waveform_underlq80

    return lq80_region

# calculate tail charge difference
def calculate_tail_charge_diff(raw_waveform, daq_energy, peak_index):
    """
    Calculates the Energy-Normalized Late Charge Residual (tail_charge_diff).
    
    tail_charge_diff = (Area_Late - Area_Early) / DAQ_Energy
    
    Args:
        raw_waveform (np.array)
        daq_energy (float): The raw energy proxy (DAQ Energy).
        peak_index (int): The index of the waveform's maximum amplitude.

    Returns:
        float: The calculated tail_charge_diff parameter.
    """
    
    # Apply Pole-Zero Correction
    pz_corrected_wf, _ = pole_zero_correction(raw_waveform)

    # Assuming 50 samples/µs
    SAMPLES_PER_MICROSECOND = 50 

    # Define Time Windows (Indices relative to peak)
    
    # Early Tail Window (0.5 µs to 1.5 µs after peak)
    T_START_EARLY_US = 0.5  
    T_END_EARLY_US   = 1.5
    
    # Late Tail Window (2.0 µs to 3.0 µs after peak)
    T_START_LATE_US  = 2.0
    T_END_LATE_US    = 3.0 
    
    # Convert µs times to array indices
    idx_start_early = peak_index + int(T_START_EARLY_US * SAMPLES_PER_MICROSECOND)
    idx_end_early   = peak_index + int(T_END_EARLY_US * SAMPLES_PER_MICROSECOND)   
    idx_start_late  = peak_index + int(T_START_LATE_US * SAMPLES_PER_MICROSECOND)
    idx_end_late    = peak_index + int(T_END_LATE_US * SAMPLES_PER_MICROSECOND)
    
    # Basic bounds check
    if idx_end_late >= len(pz_corrected_wf) or idx_start_early >= idx_end_early:
        return 0.0

    # Calculate Integrated Charge (Area)
    # Summing ADC counts is the integral (Area) over the indices.
    charge_early = np.sum(pz_corrected_wf[idx_start_early:idx_end_early])
    charge_late = np.sum(pz_corrected_wf[idx_start_late:idx_end_late])

    # Calculate Final Normalized Residual
    charge_residual = charge_late - charge_early
    
    if daq_energy > 0:
        tail_charge_diff = charge_residual / daq_energy
    else:
        tail_charge_diff = 0.0
        
    return tail_charge_diff

#  Tail Flattening Ratio (TFR)
def compute_tfr(wf_raw, wf_pz, peak_idx, tail_len=1000):
    tail_raw = wf_raw[peak_idx:peak_idx + tail_len]
    tail_pz = wf_pz[peak_idx:peak_idx + tail_len]
    
    tfr = np.std(tail_raw) / np.std(tail_pz)
    return tfr

def extract_avse(wf, E, dt=4, window_ns=100, sg_window=11, sg_poly=3):
    """
    Extract AvsE (Amplitude vs Energy) from a raw waveform.
    
    Steps:
      1. Normalize waveform to [0, 1]
      2. Smooth waveform using Savitzky-Golay filter
      3. Compute current waveform using sliding window derivative
      4. Extract peak current amplitude (A)
      5. Compute total energy (E) as area under original waveform
      6. Return AvsE = A / E

    Parameters:
        wf : 1D np.array
            Input waveform (ADC values)
        E : float
            Total energy (from DAQ)
        dt : float
            Sampling period in ns (default 10 ns)
        window_ns : float
            Window size for slope (default 100 ns)
        sg_window : int
            Window length for Savitzky-Golay filter (must be odd)
        sg_poly : int
            Polynomial order for Savitzky-Golay filter

    Returns:
        AvsE : float
            Amplitude vs Energy parameter
        A : float
            Peak current amplitude (normalized units/ns)
        E : float
            Total energy (from DAQ)
        slopes : np.array
            Current waveform (slope at each time index)
        wnorm_smooth : np.array
            Smoothed normalized waveform
    """
    # Normalize waveform to [0, 1]
    wmin, wmax = np.min(wf), np.max(wf)
    w_norm = (wf - wmin) / (wmax - wmin)

    # Smooth waveform using Savitzky-Golay filter
    wnorm_smooth = savgol_filter(w_norm, sg_window, sg_poly)

    # Compute current waveform using sliding window derivative
    N = int(window_ns / dt)  # number of samples in the window
    slopes = np.zeros(len(wnorm_smooth) - N)
    for i in range(len(slopes)):
        slopes[i] = (wnorm_smooth[i + N] - wnorm_smooth[i]) / (N * dt)

    # Peak current amplitude (A)
    A = np.max(slopes)

    # Compute AvsE
    AvsE = A / E

    return AvsE, A, E, slopes, wnorm_smooth

def gradient_trace(wf):
    """
    First difference (slope) as a simple current proxy.
    Returns the slope series and a few summary stats.
    """
    g = np.gradient(wf)
    stats = {
        "max_grad": np.max(g),
        "rms_grad": np.sqrt(np.mean(g**2)),
        "grad_spread": np.percentile(np.abs(g), 95) - np.percentile(np.abs(g), 5)
    }
    return g, stats

def compare_transforms1(wf_sse, wf_mse, time_index, smooth_sigma=1.0):
    # ---------- 2) Gradient (slope) ----------
    g_sse, stats_sse = gradient_trace(wf_sse)
    g_mse, stats_mse = gradient_trace(wf_mse)

    wf_norm_sse = (g_sse - np.min(g_sse)) / (np.max(g_sse) - np.min(g_sse))
    wf_norm_mse = (g_mse - np.min(g_mse)) / (np.max(g_mse) - np.min(g_mse))


    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10)) # Adjust figsize as needed
    axes[0].plot(time_index, wf_norm_sse, label='SSE-like Waveform', color='orange')    
    axes[0].set_title("Gradient (slope) vs time")
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Slope (ΔADC / sample)")
    axes[0].legend()

    axes[1].plot(time_index, wf_norm_mse, label='MSE-like Waveform', color='blue')
    axes[1].set_title("Gradient (slope) vs time")
    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel("Slope (ΔADC / sample)")
    axes[1].legend()
    plt.show()

# compare_transforms(true_waveform, false_waveform, time_index)

def compare_transforms2(wf_sse, wf_mse, time_index, smooth_sigma=1.0):
    # ---------- 2) Gradient (slope) ----------
    g_sse, stats_sse = gradient_trace(wf_sse)
    g_mse, stats_mse = gradient_trace(wf_mse)

    wf_norm_sse = (g_sse - np.min(g_sse)) / (np.max(g_sse) - np.min(g_sse))
    wf_norm_mse = (g_mse - np.min(g_mse)) / (np.max(g_mse) - np.min(g_mse))

    plt.figure(figsize=(20, 15))
    plt.plot(time_index, wf_norm_sse, label='SSE-like Waveform', color='orange')
    plt.plot(time_index, wf_norm_mse, label='MSE-like Waveform', color='blue', alpha = 0.3)  
    plt.title("Gradient (slope) vs time")
    plt.xlabel("Sample")
    plt.ylabel("Slope (ΔADC / sample)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# compare_transforms(true_waveform, false_waveform, time_index)