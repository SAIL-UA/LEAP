import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def create_dataframes(input_array):
    """
    Splits input feature array into two dataframes:
    - One excluding rows where the second last dimension is '3'.
    - One modifying '3' to '1' for injured vs healthy classification.

    Parameters:
    - input_array (numpy.ndarray): The feature matrix array.

    Returns:
    - tuple: (df_left_vs_right, df_injured_vs_healthy)
    """
    input_array_no_last_dim = input_array[:, :-1]
    last_column = input_array[:, -1]

    # Exclude rows where the second last dimension is '3'
    mask_df1 = input_array_no_last_dim[:, -1] != '3'
    df1 = pd.DataFrame(input_array_no_last_dim[mask_df1])
    df1['label'] = last_column[mask_df1]

    # Modify values for injured vs healthy classification
    df2_modified = input_array_no_last_dim.copy()
    df2_modified[:, -1] = np.where(df2_modified[:, -1] == '3', '1', '0')
    df2 = pd.DataFrame(df2_modified)
    df2['label'] = last_column

    return df1, df2

def evaluate_and_plot(feature_matrices):
    """
    Trains and evaluates multiple classifiers on different feature matrices, 
    then plots performance metrics for comparison.

    Parameters:
    - feature_matrices (dict): Dictionary with frequency keys (8Hz, 16Hz, 32Hz) and DataFrame values.

    Returns:
    - dict: Dictionary containing trained classifiers for each frequency.
    """
    
    classifiers = {
        "SVM": SVC(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier(),
        "Neural Network": MLPClassifier(max_iter=1000)
    }

    results = {}
    trained_models = {}

    for freq, df in feature_matrices.items():
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        results[freq] = {}
        trained_models[freq] = {}

        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            results[freq][name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted'),
                "recall": recall_score(y_test, y_pred, average='weighted'),
                "f1_score": f1_score(y_test, y_pred, average='weighted')
            }

            trained_models[freq][name] = clf  # Store trained classifier

    results_df = {freq: pd.DataFrame(res).T for freq, res in results.items()}

    # Plot Performance Metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    for i, metric in enumerate(metrics):
        ax = axs[i//2, i%2]
        for freq, df in results_df.items():
            df[metric].plot(kind='line', ax=ax, marker='o', linestyle='-', label=f'{freq} Hz')
        
        ax.set_title(metric.capitalize())
        ax.set_xlabel('Classifier')
        ax.set_ylabel(metric.capitalize())
        ax.legend()

    plt.tight_layout()
    plt.show()

    return trained_models

def train_and_save_model(phase, class1, class0, save_path, frequency_thresholds, classifier_name):
    """
    Loads feature matrix, trains models, plots evaluation, and saves the selected trained model.

    Parameters:
    - phase (str): The movement phase (e.g., 'Walk', 'Jog').
    - class1 (str): Label for class 1 (e.g., "Left").
    - class0 (str): Label for class 0 (e.g., "Right").
    - save_path (str): Directory where feature matrices are stored.
    - frequency_thresholds (list): List of frequency thresholds to process (e.g., [8, 16, 32]).
    - classifier_name (str): Classifier to train and save (e.g., "KNN", "Random Forest").

    Returns:
    - str: Path of the saved model file.
    """
    
    feature_matrices = {}

    for freq in frequency_thresholds:
        feature_file = os.path.join(save_path, f"freq_{freq}_{class1.lower()}_{class0.lower()}_{phase.lower()}.npy")

        if not os.path.exists(feature_file):
            print(f"Feature file not found: {feature_file}")
            continue

        # Load feature matrix and prepare data
        feature_data = np.load(feature_file, allow_pickle=True)
        df_left_vs_right, _ = create_dataframes(feature_data)

        feature_matrices[freq] = df_left_vs_right

    if not feature_matrices:
        raise FileNotFoundError("No valid feature matrices found. Check file paths.")

    # Train models, evaluate, and plot
    trained_models = evaluate_and_plot(feature_matrices)

    if classifier_name not in trained_models[frequency_thresholds[0]]:
        raise ValueError(f"Classifier '{classifier_name}' not supported. Choose from {list(trained_models[frequency_thresholds[0]].keys())}.")

    # Save trained model
    best_model = trained_models[frequency_thresholds[0]][classifier_name]
    model_filename = os.path.join(save_path, f"best_{classifier_name.lower()}_model_{phase.lower()}_{class1.lower()}_vs_{class0.lower()}_{frequency_thresholds[0]}hz.pkl")
    joblib.dump(best_model, model_filename)

    print(f"Trained model saved at: {model_filename}")
    return model_filename
