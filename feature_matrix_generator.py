import os
import numpy as np
from sensor_data_processor import process_multiple_participants
from window_processor import get_window_numbers_and_labels

def generate_full_paths(parent_path, finished_participants):
    """
    Generates full paths for finished participants.

    Parameters:
    - parent_path (str): The base directory containing participant folders.
    - finished_participants (list): List of participant IDs.

    Returns:
    - list: Full paths to each participant's folder.
    """
    return [os.path.join(parent_path, participant) for participant in finished_participants]

def generate_feature_matrix(parent_path, phase, increment, class1, class0, save_path, frequency_thresholds):
    """
    Generates and saves the feature matrix and raw data for different frequency thresholds.

    Parameters:
    - parent_path (str): Path to the parent directory containing participant data.
    - phase (str): The movement phase (e.g., 'Walk', 'Jog').
    - increment (int): Time window increment.
    - class1 (str): Label for class 1 (e.g., "Left").
    - class0 (str): Label for class 0 (e.g., "Right").
    - save_path (str): Directory where processed feature matrices should be saved.
    - frequency_thresholds (list): List of frequency thresholds to process (e.g., [8, 16, 32]).

    Returns:
    - dict: Paths of saved `.npy` files.
    """

    # Ensure the save path exists
    os.makedirs(save_path, exist_ok=True)

    # List participant paths
    finished_participants = [p for p in os.listdir(parent_path) if os.path.isdir(os.path.join(parent_path, p))]
    full_paths = generate_full_paths(parent_path, finished_participants)

    # Get window start/end points and labels
    starts, ends, labels, missing_sensors_participants, finished_participants, total_completed, total_missing_sensors, missing_percentage, skipped_participants = get_window_numbers_and_labels(
        full_paths, phase, finished_participants, left_label=class1, right_label=class0
    )

    saved_files = {}

    for freq in frequency_thresholds:
        freq_matrix, raw_data = process_multiple_participants(full_paths, phase, starts, ends, increment, labels, freq)

        freq_filename = f"freq_{freq}_{class1.lower()}_{class0.lower()}_{phase.lower()}.npy"
        data_filename = f"data_{freq}_{class1.lower()}_{class0.lower()}_{phase.lower()}.npy"

        np.save(os.path.join(save_path, freq_filename), freq_matrix)
        np.save(os.path.join(save_path, data_filename), raw_data)

        saved_files[freq] = {
            "feature_matrix": os.path.join(save_path, freq_filename),
            "raw_data": os.path.join(save_path, data_filename),
        }

    return saved_files
