# LEAP

LEAP is a Python package designed for processing and analyzing sensor data, particularly in the context of movement phases such as walking or jogging. The package provides tools for generating feature matrices, training machine learning models, evaluating feature importance, and visualizing data using t-SNE plots.

## Table of Contents

- [Installation](#installation)
- [Modules and Functions](#modules-and-functions)
  - [Feature Importance Analyzer](#feature-importance-analyzer)
  - [Feature Matrix Generator](#feature-matrix-generator)
  - [Ground Truth Processor](#ground-truth-processor)
  - [Model Trainer](#model-trainer)
  - [Sensor Data Processor](#sensor-data-processor)
  - [t-SNE Visualizer](#t-sne-visualizer)
  - [Window Processor](#window-processor)
- [Usage Examples](#usage-examples)
- [License](#license)

## Installation

To install the LEAP package, clone the repository and install the required dependencies:

```bash
git clone https://github.com/SAIL-UA/LEAP.git
cd LEAP
pip install -r requirements.txt
```

## Modules and Functions

### Feature Importance Analyzer

**File**: `feature_importance_analyzer.py`

This module calculates and visualizes permutation importance of features using trained models.

#### `analyze_permutation_importance(model_path, feature_matrix, label, save_csv_path, heatmap_title)`

**Parameters:**
- `model_path` (str): Path to the saved model file.
- `feature_matrix` (pd.DataFrame): Feature matrix.
- `label` (str): Label for the target column.
- `save_csv_path` (str): Path to save the importance CSV.
- `heatmap_title` (str): Title for the heatmap.

### Feature Matrix Generator

**File**: `feature_matrix_generator.py`

This module generates feature matrices from sensor data.

#### `generate_feature_matrix(parent_path, phase, increment, class1, class0, save_path, frequency_thresholds)`

**Parameters:**
- `parent_path` (str): Path to participant data.
- `phase` (str): Movement phase (e.g., 'Walk').
- `increment` (int): Time window increment.
- `class1` (str): Label for class 1.
- `class0` (str): Label for class 0.
- `save_path` (str): Directory to save matrices.
- `frequency_thresholds` (list): Frequency thresholds to process.

### Ground Truth Processor

**File**: `ground_truth_processor.py`

This module assigns ground truth labels based on demographic data.

#### `generate_ground_truth(demographic_file, parent_folder)`

**Parameters:**
- `demographic_file` (str): Path to Excel file with demographics.
- `parent_folder` (str): Path to participant folders.

### Model Trainer

**File**: `model_trainer.py`

This module trains and evaluates classifiers on feature matrices.

#### `train_and_save_model(phase, class1, class0, save_path, frequency_thresholds, classifier_name)`

**Parameters:**
- `phase` (str): Movement phase.
- `class1` (str): Label for class 1.
- `class0` (str): Label for class 0.
- `save_path` (str): Directory for saving models.
- `frequency_thresholds` (list): Frequency thresholds.
- `classifier_name` (str): Classifier to train and save.

### Sensor Data Processor

**File**: `sensor_data_processor.py`

Processes raw sensor data to compute Phase Slope Index (PSI) features.

#### `process_all_windows(base_path, phase, start, end, increment, freq)`

**Parameters:**
- `base_path` (str): Directory of sensor data.
- `phase` (str): Movement phase.
- `start` (int): Start timestamp.
- `end` (int): End timestamp.
- `increment` (int): Window increment.
- `freq` (int): Max frequency for PSI.

### t-SNE Visualizer

**File**: `tsne_visualizer.py`

Visualizes feature data using t-SNE plots.

#### `plot_tsne(feature_matrix, model_path, label_description, perplexity=30)`

**Parameters:**
- `feature_matrix` (pd.DataFrame or np.ndarray): Feature matrix.
- `model_path` (str): Path to trained model.
- `label_description` (dict): Dictionary mapping labels.
- `perplexity` (int): t-SNE perplexity parameter.

### Window Processor

**File**: `window_processor.py`

Extracts window numbers and labels from sensor data.

#### `get_window_numbers_and_labels(base_paths, phase, complete_participants, left_label='Left', right_label='Right')`

**Parameters:**
- `base_paths` (list): List of participant paths.
- `phase` (str): Movement phase.
- `complete_participants` (list): Participants to process.
- `left_label` (str): Label for left-injured participants.
- `right_label` (str): Label for right-injured participants.

## Usage Examples

### Load Sensor Data

```python
from leap import load_data

data = load_data(base_path='path/to/data', phase='Walking', window='60s')
print(data)
```

### Generate Feature Matrix

```python
from leap import generate_feature_matrix

saved_files = generate_feature_matrix(parent_path='path/to/data', phase='Walk', increment=600, class1='Left', class0='Right', save_path='path/to/save', frequency_thresholds=[8, 16, 32])
print(saved_files)
```

### Train and Save a Model

```python
from leap import train_and_save_model

model_path = train_and_save_model(phase='Walk', class1='Left', class0='Right', save_path='path/to/save', frequency_thresholds=[8, 16, 32], classifier_name='KNN')
print(model_path)
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

