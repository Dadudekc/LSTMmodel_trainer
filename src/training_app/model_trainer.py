# src/training_app/model_trainer.py

import os
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

# Setup Logging
logging.basicConfig(level=logging.INFO)

class ModelTrainer:
    def __init__(self, dataset_path, target_column, model_type, hyperparameters):
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.model_type = model_type
        self.hyperparameters = hyperparameters
        self.model = None
        self.metrics = {}
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.predictions = None

    def load_data(self):
        logging.info("Loading data from dataset.")
        if self.dataset_path.endswith('.csv'):
            data = pd.read_csv(self.dataset_path)
        else:
            data = pd.read_excel(self.dataset_path)
        if self.target_column not in data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset.")
        logging.info("Data loaded successfully.")
        return data

    def preprocess_data(self, data):
        logging.info("Preprocessing data.")
        # Handle missing values
        data = data.dropna()
        logging.info(f"Data shape after dropping NA: {data.shape}")

        # Separate features and target
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]

        # Encode categorical variables if any
        X = pd.get_dummies(X, drop_first=True)
        logging.info(f"Features shape after encoding: {X.shape}")

        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logging.info("Feature scaling completed.")

        return X_scaled, y

    def select_model(self):
        logging.info(f"Selecting model type: {self.model_type}")
        if self.model_type == 'linear_regression':
            self.model = LinearRegression(**self.hyperparameters)
        elif self.model_type == 'random_forest_classifier':
            self.model = RandomForestClassifier(**self.hyperparameters)
        elif self.model_type == 'random_forest_regressor':
            self.model = RandomForestRegressor(**self.hyperparameters)
        elif self.model_type == 'svm_classifier':
            self.model = SVC(**self.hyperparameters)
        elif self.model_type == 'svm_regressor':
            self.model = SVR(**self.hyperparameters)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        logging.info(f"Model {self.model_type} initialized with hyperparameters: {self.hyperparameters}")

    def train(self, X_train, y_train):
        logging.info("Starting model training.")
        self.model.fit(X_train, y_train)
        logging.info("Model training completed.")

    def evaluate(self, X_test, y_test):
        logging.info("Evaluating model.")
        self.predictions = self.model.predict(X_test)
        self.y_test = y_test  # Store for confusion matrix
        if self.model_type in ['linear_regression', 'random_forest_regressor', 'svm_regressor']:
            self.metrics['Mean Squared Error'] = mean_squared_error(y_test, self.predictions)
            self.metrics['R² Score'] = r2_score(y_test, self.predictions)
            logging.info("Regression metrics calculated.")
        else:
            self.metrics['Accuracy'] = accuracy_score(y_test, self.predictions)
            self.metrics['Precision'] = precision_score(y_test, self.predictions, average='weighted', zero_division=0)
            self.metrics['Recall'] = recall_score(y_test, self.predictions, average='weighted', zero_division=0)
            self.metrics['F1 Score'] = f1_score(y_test, self.predictions, average='weighted', zero_division=0)
            self.metrics['Confusion Matrix'] = confusion_matrix(y_test, self.predictions).tolist()
            self.metrics['Classes'] = sorted(list(set(y_test)))
            logging.info("Classification metrics calculated.")

    def cross_validate_model(self, X, y, cv=5):
        logging.info("Starting cross-validation.")
        if self.model_type in ['linear_regression', 'random_forest_regressor', 'svm_regressor']:
            scores = cross_val_score(self.model, X, y, cv=cv, scoring='r2')
            self.metrics['Cross-Validation R² Mean'] = scores.mean()
            self.metrics['Cross-Validation R² Std'] = scores.std()
            logging.info("Cross-validation for regression completed.")
        else:
            scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
            self.metrics['Cross-Validation Accuracy Mean'] = scores.mean()
            self.metrics['Cross-Validation Accuracy Std'] = scores.std()
            logging.info("Cross-validation for classification completed.")

    def save_model(self, save_path):
        logging.info(f"Saving model to {save_path}.")
        joblib.dump(self.model, save_path)
        logging.info("Model saved successfully.")

    def load_model(self, model_path):
        logging.info(f"Loading model from {model_path}.")
        self.model = joblib.load(model_path)
        logging.info("Model loaded successfully.")

    def run(self):
        """
        Executes the full training pipeline:
        1. Load data
        2. Preprocess data
        3. Split data
        4. Select model
        5. Train model
        6. Evaluate model
        7. Cross-validate model
        Returns:
            dict: Evaluation metrics
        """
        data = self.load_data()
        X, y = self.preprocess_data(data)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.select_model()
        self.train(self.X_train, self.y_train)
        self.evaluate(self.X_test, self.y_test)
        self.cross_validate_model(X, y)
        logging.info("Training pipeline executed successfully.")
        return self.metrics
