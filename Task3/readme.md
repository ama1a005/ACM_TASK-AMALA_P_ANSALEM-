
# Breast Cancer Classification using Neural Networks

## Overview
This project uses a neural network model to classify breast cancer as benign or malignant. The code is built using TensorFlow and Keras for model creation and training, and scikit-learn for dataset handling and evaluation metrics.

## Dataset
The model uses the Breast Cancer dataset from scikit-learn, which contains 569 samples and 30 features. The target variable indicates whether the cancer is benign (`1`) or malignant (`0`).

## Code Structure
1. **Data Preprocessing**: The dataset is loaded, and features are standardized using `StandardScaler`. The data is split into training and testing sets.
2. **Model Definition**: A neural network is built with 3 layers:
   - 16 neurons in the first hidden layer.
   - 8 neurons in the second hidden layer.
   - 1 output neuron with sigmoid activation for binary classification.
3. **Model Training**: The model is compiled using `Adam` optimizer and trained for 25 epochs.
4. **Evaluation**: The model is evaluated on the test set using accuracy, precision, recall, and F1-score.
5. **Visualization**: Plots for training and validation accuracy/loss over epochs are generated using `matplotlib`.

## Results
The model achieves good performance metrics, demonstrating its ability to classify breast cancer cases effectively.

## Requirements
