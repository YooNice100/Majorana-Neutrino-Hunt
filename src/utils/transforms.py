import numpy as np
from scipy.optimize import curve_fit
from scipy.fft import rfft, rfftfreq



def estimate_baseline(y, n_samples=200):
    """
    Returns baseline (mean, std) from first n_samples.
    """
    y0 = np.asarray(y, dtype=float)[:n_samples]
    return float(np.mean(y0)), float(np.std(y0))



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



def compute_frequency_spectrum(waveform, sample_spacing=1.0):
    """
    Computes the one-sided frequency spectrum of a real waveform.

    Parameters:
        waveform (np.ndarray): 1D array of samples
        sample_spacing (float): time between samples

    Returns:
        xf (np.ndarray): frequency values
        amplitude (np.ndarray): magnitude of spectrum
    """
    N = len(waveform)
    yf = rfft(waveform)
    xf = rfftfreq(N, d=sample_spacing)
    amplitude = np.abs(yf) * 2 / N
    return xf, amplitude