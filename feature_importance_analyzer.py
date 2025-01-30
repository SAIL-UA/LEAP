import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance

# Define sensors and dimensions
sensors = ['Left_Ankle', 'Right_Ankle', 'Left_Wrist', 'Right_Wrist', 'Sacrum']
dimensions = ['x_acc', 'y_acc', 'z_acc', 'x_gyro', 'y_gyro', 'z_gyro']
num_sensors = len(sensors)
num_dimensions = len(dimensions)

# Generate feature indices
def get_feature_indices(num_sensors, num_dimensions):
    feature_indices = []
    for i in range(num_sensors * num_dimensions):
        for j in range(i + 1, num_sensors * num_dimensions):
            feature_indices.append((i, j))
    return feature_indices

feature_indices = get_feature_indices(num_sensors, num_dimensions)

# Mapping function to correctly map features to sensor pairs and dimensions
def map_feature_to_sensor_pair(feature_idx):
    i, j = feature_indices[feature_idx]
    sensor1, dim1 = divmod(i, num_dimensions)
    sensor2, dim2 = divmod(j, num_dimensions)
    return (sensors[sensor1], sensors[sensor2]), (dimensions[dim1], dimensions[dim2])

# Function to calculate permutation importance for the entire dataset
def calculate_global_importance(X, y, model, save_csv_path, n_repeats=10, n_jobs=-1):
    """
    Computes the permutation importance of features using a trained model.

    Parameters:
    - X (numpy.ndarray): Feature matrix.
    - y (numpy.ndarray): Labels.
    - model (sklearn model): Trained classifier model.
    - save_csv_path (str): Path to save the importance values in CSV format.
    - n_repeats (int): Number of permutations for importance calculation.
    - n_jobs (int): Number of parallel jobs.

    Returns:
    - DataFrame: Importance data sorted by significance.
    """
    
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=n_jobs)
    
    importances = result.importances_mean
    feature_names = ['feature_{}'.format(i) for i in range(X.shape[1])]
    
    sorted_indices = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_indices]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]
    
    important_sensor_pairs = []
    for feature, importance in zip(sorted_feature_names, sorted_importances):
        feature_idx = int(feature.split('_')[1])
        sensor_pair, dimensions = map_feature_to_sensor_pair(feature_idx)
        important_sensor_pairs.append((sensor_pair, dimensions, importance))

    # Convert the results to a DataFrame and save
    df_importance = pd.DataFrame(important_sensor_pairs, columns=['Pair', 'Dimensions', 'Importance'])
    df_importance.to_csv(save_csv_path, index=False)

    return df_importance

# Function to generate a heatmap of sensor pair importance
def plot_sensor_pair_importance(csv_file, title):
    """
    Plots a heatmap of sensor pair importance based on the input CSV file.

    Parameters:
    - csv_file (str): Path to the CSV file containing sensor pair importance data.
    - title (str): Title of the heatmap plot.
    """
    # Load the importance data
    importance_data = pd.read_csv(csv_file)

    # Ignore negative importance values
    importance_data = importance_data[importance_data['Importance'] > 0]

    # Create a dictionary to store summed importance values for each sensor pair
    pair_importance = {sensor: {sensor: 0 for sensor in sensors} for sensor in sensors}

    # Sum the importance values for each pair of sensors
    for _, row in importance_data.iterrows():
        sensor1, sensor2 = eval(row['Pair'])  # Convert string to tuple
        importance = row['Importance']
        pair_importance[sensor1][sensor2] += importance
        pair_importance[sensor2][sensor1] += importance  # Ensure symmetry

    # Prepare data for the heatmap
    heatmap_data = np.zeros((len(sensors), len(sensors)))

    for i, sensor1 in enumerate(sensors):
        for j, sensor2 in enumerate(sensors):
            heatmap_data[i, j] = pair_importance[sensor1][sensor2]

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='coolwarm', cbar=True, square=True, linewidths=0.5, xticklabels=sensors, yticklabels=sensors)

    # Customize the plot
    plt.title(title)
    plt.show()

def analyze_permutation_importance(model_path, feature_matrix, label, save_csv_path, heatmap_title):
    """
    Loads a trained model, computes permutation importance, saves the results, and generates a heatmap.

    Parameters:
    - model_path (str): Path to the saved model file.
    - feature_matrix (pd.DataFrame): Feature matrix.
    - label (str): Label for the target column.
    - save_csv_path (str): Path to save the importance CSV.
    - heatmap_title (str): Title for the heatmap.

    Returns:
    - None (Plots heatmap and prints summary).
    """
    
    # Load trained model
    model = joblib.load(model_path)

    # Extract features and labels
    X = feature_matrix.iloc[:, :-1].values
    y = feature_matrix[label].values

    # Calculate permutation importance and save results
    df_importance = calculate_global_importance(X, y, model, save_csv_path)

    print(f"Permutation importance saved to: {save_csv_path}")

    # Generate heatmap
    plot_sensor_pair_importance(save_csv_path, heatmap_title)
