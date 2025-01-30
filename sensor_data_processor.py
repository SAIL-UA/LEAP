import os
import numpy as np
import pandas as pd
from numpy.fft import fft
from numpy import conj, arctan2

from data_loader import load_data # Ensure `load_data` is available in the package


class SensorDataProcessor:
    def __init__(self, window_size=600, sampling_rate=60, max_freq=8):
        """
        Initializes the sensor data processor.

        Parameters:
        - window_size (int): Number of samples per window.
        - sampling_rate (int): Frequency of data collection in Hz.
        - max_freq (int): Maximum frequency to analyze for PSI computation.
        """
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.max_freq = max_freq

    def calculate_psi(self, fft1, fft2):
        """
        Computes the Phase Slope Index (PSI) between two signals.

        Parameters:
        - fft1 (numpy.ndarray): FFT-transformed signal from sensor 1.
        - fft2 (numpy.ndarray): FFT-transformed signal from sensor 2.

        Returns:
        - float: The PSI value (slope of the phase spectrum).
        """
        csd = self.cross_spectral_density(fft1, fft2)
        phase_spec = self.phase_spectrum(csd)

        # Determine frequency resolution and valid index range
        freq_resolution = self.sampling_rate / self.window_size
        max_index = int(self.max_freq / freq_resolution)

        freq_indices = np.arange(0, max_index + 1)
        phase_values = phase_spec[0:max_index + 1]

        # Linear regression to estimate the PSI slope
        slope, _ = np.polyfit(freq_indices, phase_values, 1)
        return slope

    def cross_spectral_density(self, fft1, fft2):
        """Computes Cross-Spectral Density."""
        return fft1 * conj(fft2)

    def phase_spectrum(self, csd):
        """Computes the phase spectrum from the cross-spectral density."""
        return arctan2(csd.imag, csd.real)

    def process_data(self, sensor_data):
        """
        Computes PSI for all sensor pairs and dimensions.

        Parameters:
        - sensor_data (list): A list of sensor data arrays.

        Returns:
        - numpy.ndarray: Flattened array of PSI values.
        """
        num_sensors = len(sensor_data)
        num_dimensions = sensor_data[0].shape[1]
        psi_matrix = np.zeros((num_sensors * num_dimensions, num_sensors * num_dimensions))

        for i in range(num_sensors):
            for j in range(i + 1, num_sensors):
                for dim1 in range(num_dimensions):
                    for dim2 in range(num_dimensions):
                        fft1 = fft(sensor_data[i][:, dim1])
                        fft2 = fft(sensor_data[j][:, dim2])
                        psi_value = self.calculate_psi(fft1, fft2)
                        psi_matrix[i * num_dimensions + dim1, j * num_dimensions + dim2] = psi_value
                        psi_matrix[j * num_dimensions + dim2, i * num_dimensions + dim1] = -psi_value

        # Extract upper-triangle (unique PSI values)
        return psi_matrix[np.triu_indices(num_sensors * num_dimensions, k=1)]


def get_feature_indices(num_sensors, num_dimensions):
    """Returns indices of PSI feature pairs."""
    return [(i, j) for i in range(num_sensors * num_dimensions) for j in range(i + 1, num_sensors * num_dimensions)]


def process_all_windows(base_path, phase, start=0, end=3000, increment=600, freq=8):
    """
    Processes multiple time windows and computes PSI features.

    Parameters:
    - base_path (str): Directory where sensor data is stored.
    - phase (str): Name of the movement phase (e.g., 'Walking', 'Jogging').
    - start (int): Starting timestamp for processing.
    - end (int): End timestamp.
    - increment (int): Window increment step.
    - freq (int): Maximum frequency to analyze for PSI.

    Returns:
    - tuple: (all PSI features as numpy array, all raw sensor data as numpy array)
    """
    processor = SensorDataProcessor(max_freq=freq)
    all_features, all_sensor_data = [], []

    for window in range(start, end + 1, increment):
        window_str = str(window)
        try:
            sensor_data = load_data(base_path, phase, window_str)
            features = processor.process_data(sensor_data)
            all_features.append(features)
            all_sensor_data.append(sensor_data)
        except FileNotFoundError as e:
            print(f"Skipping window {window}: {e}")

    return np.array(all_features), np.array(all_sensor_data)


def process_participant(base_path, phase, start, end, increment, label, freq):
    """Processes PSI features for a single participant."""
    features, data = process_all_windows(base_path, phase, start, end, increment, freq)
    num_windows = features.shape[0]
    basename = os.path.basename(base_path)
    basenames_column = np.full((num_windows, 1), basename)
    labels = np.full((num_windows, 1), label)
    return np.hstack((features, labels, basenames_column)), data


def process_multiple_participants(base_paths, phase, starts, ends, increment, labels, freq):
    """Processes PSI features for multiple participants."""
    all_features, all_data = [], []
    for base_path, start, end, label in zip(base_paths, starts, ends, labels):
        features_with_labels, data = process_participant(base_path, phase, start, end, increment, label, freq)
        all_features.append(features_with_labels)
        all_data.append(data)

    return np.concatenate(all_features, axis=0), np.concatenate(all_data, axis=0)
