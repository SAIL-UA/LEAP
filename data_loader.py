import os
import pandas as pd

def load_data(base_path, phase, window):
    """
    Loads sensor data from a specified directory structure.

    Parameters:
    - base_path (str): The base directory where sensor data is stored.
    - phase (str): The phase name to look for (e.g., 'Walking', 'Jogging').
    - window (str): The window size identifier for the file (e.g., '60s').

    Returns:
    - list: A list of NumPy arrays containing accelerometer and gyroscope data from different sensors.

    Raises:
    - FileNotFoundError: If the required folders or files are missing.
    """
    sensors = ['LeftAnkle', 'RightAnkle', 'LeftWrist', 'RightWrist', 'Sacrum']
    data = []

    for sensor in sensors:
        # Locate sensor folder
        sensor_folder = next(
            (os.path.join(base_path, folder) for folder in os.listdir(base_path) if sensor in folder),
            None
        )

        if sensor_folder is None:
            raise FileNotFoundError(f'Folder for sensor {sensor} not found in {base_path}')

        # Locate phase folder
        phase_folder = next(
            (os.path.join(sensor_folder, folder) for folder in os.listdir(sensor_folder) if phase.lower() in folder.lower()),
            None
        )

        if phase_folder is None:
            raise FileNotFoundError(f'Folder for phase {phase} not found in {sensor_folder}')

        # Locate the correct CSV file
        file_path = next(
            (os.path.join(phase_folder, file) for file in os.listdir(phase_folder) if file.endswith(f'{window}.csv')),
            None
        )

        if file_path is None or not os.path.exists(file_path):
            raise FileNotFoundError(f'File ending with _{window}.csv not found in {phase_folder}')

        # Read CSV file and extract relevant columns
        df = pd.read_csv(file_path)
        sensor_data = df.iloc[:, [1, 2, 3, 7, 8, 9]].values  # Columns: [Acc_X, Acc_Y, Acc_Z, Gyro_X, Gyro_Y, Gyro_Z]
        data.append(sensor_data)

    return data
