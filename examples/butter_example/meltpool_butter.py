import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter,butter

class MeltPoolFiltration:
    def __init__(self,
                 mu: float,
                 sigma_total: float,
                 scan_speed: float,
                 timeseries_params: list):
        """
        Filtration of disparate fluctuation scales to infer a melt pool oscillations sequence.
        Uses scipy.butter to convolve scales of fluctuations together.
        Initialization parameters:
        mu: mean melt pool dimension
        sigma_total: target total standard deviation of the fluctuations
        scan_speed: speed in m/s
        timeseries_params: list of [fs,duration]
        """
        # statistical properties
        self.mu,self.sigma_total = mu,sigma_total
        # process parameters
        self.scan_speed = scan_speed
        # timeseries related properties
        self.fs,self.duration = timeseries_params
        self.dt = 1/self.fs
        self.n_points = int(self.duration/self.dt)
        self.t = np.arange(0,self.duration,self.dt)
        # parametric representations of fluctuation scales
        self.physical_effects = {} # contains scale description and parameters
    
    def add_effect(self,
                   effect_name: str,
                   effect_params: list):
        """
        Adds a physical effect {effect_name} with parameters
        length_scale_m,frequency_hz,sigma_weight = effect_params
        to the MeltPoolFiltration.physical_effects dictionary.
        """
        length_scale_m,frequency_hz,sigma_weight = effect_params
        self.physical_effects[effect_name] = {
            'length_scale_m': length_scale_m,
            'frequency_hz': frequency_hz,
            'sigma_weight': sigma_weight
        }
    
    def initialize(self,verbose=True):
        if verbose:
            print("--- Model Initialization ---")
        # Calculate frequencies from length scales
        for name, params in self.physical_effects.items():
            if params['length_scale_m'] is not None:
                params['frequency_hz'] = self.scan_speed / params['length_scale_m']
                
        # Check for Nyquist limit violations
        max_freq = max(p['frequency_hz'] for p in self.physical_effects.values())
        if max_freq > self.fs / 2:
            raise ValueError(f"Error: Maximum frequency ({max_freq/1000:.1f} kHz) exceeds Nyquist limit ({self.fs/2000:.1f} kHz). Increase sampling rate 'fs'.")
        
        # Normalize sigma weights so the variances sum correctly
        weights = np.array([p['sigma_weight'] for p in self.physical_effects.values()])
        sum_of_sq_weights = np.sum(weights**2)
        self.normalization_factor = np.sqrt(sum_of_sq_weights)
        
        for name, params in self.physical_effects.items():
            params['sigma_contribution'] = (params['sigma_weight'] / self.normalization_factor) * self.sigma_total
            if verbose:
                print(f"  > {name}:")
                print(f"    - Frequency: {params['frequency_hz']/1000:.2f} kHz")
                print(f"    - Sigma Contribution: {params['sigma_contribution']:.2f} µm")
        
    def bandpass_filter(self, data, f0, bandwidth_fraction, fs, order=4):
        """Applies a bandpass filter around a center frequency f0."""
        lowcut = f0 * (1 - bandwidth_fraction / 2)
        highcut = f0 * (1 + bandwidth_fraction / 2)
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        if high >= 1.0: high = 0.999
        if low <= 0.0001: low = 0.0001
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)

    def generate_fluctuations(self,noise_scale,verbose=True):
        base_white_noise = np.random.normal(loc=0,scale=noise_scale, size=self.n_points)
        final_series = np.zeros(self.n_points)
        self.component_series = {}

        # Create each component series, scale it, and add to the final series
        for name, params in self.physical_effects.items():
            component_noise = self.bandpass_filter(
                data=base_white_noise,
                f0=params['frequency_hz'],
                bandwidth_fraction=1,
                fs=self.fs
            )
            
            std_dev = np.std(component_noise)
            scaled_component = component_noise * (params['sigma_contribution'] / std_dev)
            self.component_series[name] = scaled_component
            final_series += scaled_component
        
        # Adding the mean
        final_series += self.mu

        if verbose:
            for name, params in self.physical_effects.items():
                sigma_contrib = (params['sigma_weight'] / self.normalization_factor) * self.sigma_total
                params['amplitude'] = sigma_contrib * np.sqrt(2) # Key deterministic step
                print(f"  > Calculated parameters for '{name}':")
                print(f"    - Frequency (fᵢ): {params['frequency_hz']/1000:.3f} kHz")
                print(f"    - Amplitude (Aᵢ): {params['amplitude']:.3f} µm")
        return final_series
        
        