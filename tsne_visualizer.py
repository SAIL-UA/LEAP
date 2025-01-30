import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap

def plot_tsne(feature_matrix, model_path, label_description, perplexity=30):
    """
    Generates t-SNE visualizations of the given feature matrix.
    
    - The first plot colors data points by actual labels.
    - The second plot colors data points by the model’s prediction confidence.

    Parameters:
    - feature_matrix (pd.DataFrame or np.ndarray): Feature matrix with labels as the last column.
    - model_path (str): Path to the trained model file (.pkl).
    - label_description (dict): Dictionary mapping labels to meaningful names.
    - perplexity (int): t-SNE perplexity parameter (default: 30).

    Returns:
    - None (Displays t-SNE plots).
    """

    # Load trained model
    knn = joblib.load(model_path)

    # Convert to DataFrame if necessary
    if isinstance(feature_matrix, np.ndarray):
        feature_matrix = pd.DataFrame(feature_matrix)

    # Separate features and labels
    X = feature_matrix.iloc[:, :-1].values
    y = feature_matrix.iloc[:, -1].values

    # Compute probabilities for the predictions
    probabilities = knn.predict_proba(X)
    max_probabilities = np.max(probabilities, axis=1)

    # Perform t-SNE on the entire dataset
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Plot the original data t-SNE (colored by actual labels)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.title('t-SNE plot of Original PSI Feature Matrix')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.grid(True)
    
    # Create legend
    handles, _ = scatter.legend_elements()
    legend_labels = [label_description.get(int(label), f'Class {label}') for label in np.unique(y)]
    plt.legend(handles, legend_labels, title='Labels', loc='best')
    plt.show()

    # Plot the t-SNE of the KNN predictions with colors based on probabilities
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=max_probabilities, cmap='viridis', alpha=0.7)
    plt.title('t-SNE plot of KNN Predictions with Probability Coloring')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.colorbar(label='Prediction Probability')
    plt.grid(True)
    plt.show()
