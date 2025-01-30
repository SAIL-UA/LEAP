import os

def get_window_numbers_and_labels(base_paths, phase, complete_participants, left_label="Left", right_label="Right"):
    """
    Extracts start and end window numbers and labels from sensor data.

    Parameters:
    - base_paths (list): List of participant folder paths.
    - phase (str): The movement phase (e.g., 'Walk').
    - complete_participants (list): List of participants to process.
    - left_label (str): Label for left-injured participants (default: "Left").
    - right_label (str): Label for right-injured participants (default: "Right").

    Returns:
    - tuple: (starts, ends, labels, missing_sensors_participants, finished_participants, 
              total_completed, total_missing_sensors, missing_percentage, skipped_participants)
    """

    starts, ends, labels = [], [], []
    missing_sensors_participants = []
    finished_participants = []
    skipped_participants = []

    for base_path in base_paths:
        participant_id = os.path.basename(base_path)

        if participant_id not in complete_participants:
            skipped_participants.append(participant_id)
            continue

        # Find Ground_Truth.txt
        ground_truth_file = next(
            (os.path.join(base_path, file) for file in os.listdir(base_path) if file.endswith('Ground_Truth.txt')),
            None
        )

        if not ground_truth_file:
            skipped_participants.append(participant_id)
            continue

        # Determine participant label
        with open(ground_truth_file, 'r') as f:
            injury_info = f.read().strip()
            if right_label in injury_info:
                label = 0
            elif left_label in injury_info:
                label = 1
            elif "Healthy" in injury_info:
                label = 3
            else:
                skipped_participants.append(participant_id)
                continue

        # Process sensor data
        sensors = ['LeftAnkle', 'RightAnkle', 'LeftWrist', 'RightWrist', 'Sacrum']
        start_window, min_end_window = None, None
        missing_sensor = False

        for sensor in sensors:
            sensor_folder = next(
                (os.path.join(base_path, folder) for folder in os.listdir(base_path) if sensor in folder),
                None
            )

            if not sensor_folder:
                missing_sensor = True
                break

            phase_folder = next(
                (os.path.join(sensor_folder, folder) for folder in os.listdir(sensor_folder) if phase.lower() in folder.lower()),
                None
            )

            if not phase_folder:
                missing_sensor = True
                break

            # Extract window numbers
            window_numbers = [
                int(file.replace('.csv', '')) for file in os.listdir(phase_folder) if file.endswith('.csv') and file.replace('.csv', '').isdigit()
            ]

            if window_numbers:
                start_window = min(start_window, min(window_numbers)) if start_window is not None else min(window_numbers)
                min_end_window = min(min_end_window, max(window_numbers)) if min_end_window is not None else max(window_numbers)
            else:
                missing_sensor = True
                break

        if missing_sensor:
            missing_sensors_participants.append(participant_id)
            continue

        starts.append(start_window)
        ends.append(min_end_window)
        labels.append(label)
        finished_participants.append(participant_id)

    total_completed = len(finished_participants)
    total_missing_sensors = len(missing_sensors_participants)
    missing_percentage = (total_missing_sensors / (total_completed + total_missing_sensors)) * 100 if (total_completed + total_missing_sensors) > 0 else 0

    return starts, ends, labels, missing_sensors_participants, finished_participants, total_completed, total_missing_sensors, missing_percentage, skipped_participants
