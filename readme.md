# 🧠 PyQt Model Trainer

## Overview

**PyQt Model Trainer** is a Python-based desktop application that allows users to load datasets, select machine learning models, configure hyperparameters, train models locally, and evaluate their performance. The application provides a user-friendly interface for performing end-to-end machine learning tasks without requiring deep technical knowledge.

## Features

- **User-Friendly Interface**
  - Intuitive GUI built with PyQt5.
  - Easy navigation through different functionalities.

- **Data Handling**
  - Load datasets in CSV and Excel formats.
  - Preview and explore data within the application.
  - Handle missing values and perform basic data preprocessing.

- **Model Selection and Configuration**
  - Choose from multiple machine learning algorithms:
    - Linear Regression
    - Random Forest Regressor
    - Random Forest Classifier
    - SVM Regressor
    - SVM Classifier
  - Configure hyperparameters for selected models through the GUI.

- **Model Training**
  - Train models locally using Scikit-learn.
  - Display training progress and status.
  - Handle long-running training processes without freezing the UI.

- **Evaluation and Metrics**
  - Compute and display standard evaluation metrics:
    - Accuracy
    - Precision
    - Recall
    - F1-Score
    - Mean Squared Error
    - R² Score
  - Visualize metrics through plots and charts.

- **Model Persistence**
  - Save trained models to disk for future use.
  - Load and manage saved models within the application.

- **Visualization**
  - Plot feature importances, confusion matrices, ROC curves, and other relevant charts.
  - Interactive plots using Matplotlib.

- **Logging and Error Handling**
  - Log training activities and errors.
  - Provide meaningful error messages to assist users.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/DaDudeKC/LSTMTrainer.git
cd PyQtModelTrainer
