import os
import json
import logging
import configparser
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
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
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import (
    StandardScaler,
    PowerTransformer,
    MinMaxScaler,
    RobustScaler,
    QuantileTransformer,
    Normalizer,
    MaxAbsScaler,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from keras_tuner import HyperModel, RandomSearch

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

    def load_data(self):
        data = pd.read_csv(self.dataset_path)
        if self.target_column not in data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset.")
        return data

    def preprocess_data(self, data):
        # Handle missing values
        data = data.dropna()

        # Separate features and target
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]

        # Encode categorical variables if any
        X = pd.get_dummies(X, drop_first=True)

        return X, y

    def select_model(self):
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

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)

        if self.model_type in ['linear_regression', 'random_forest_regressor', 'svm_regressor']:
            self.metrics['Mean Squared Error'] = mean_squared_error(y_test, predictions)
            self.metrics['R² Score'] = r2_score(y_test, predictions)
        else:
            # Assuming classification
            self.metrics['Accuracy'] = accuracy_score(y_test, predictions)
            self.metrics['Precision'] = precision_score(y_test, predictions, average='weighted', zero_division=0)
            self.metrics['Recall'] = recall_score(y_test, predictions, average='weighted', zero_division=0)
            self.metrics['F1 Score'] = f1_score(y_test, predictions, average='weighted', zero_division=0)

    def cross_validate_model(self, X, y, cv=5):
        if self.model_type in ['linear_regression', 'random_forest_regressor', 'svm_regressor']:
            scores = cross_val_score(self.model, X, y, cv=cv, scoring='r2')
            self.metrics['Cross-Validation R² Mean'] = scores.mean()
            self.metrics['Cross-Validation R² Std'] = scores.std()
        else:
            scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
            self.metrics['Cross-Validation Accuracy Mean'] = scores.mean()
            self.metrics['Cross-Validation Accuracy Std'] = scores.std()

    def save_model(self, save_path):
        joblib.dump(self.model, save_path)

    def load_model(self, model_path):
        self.model = joblib.load(model_path)

    def run(self):
        data = self.load_data()
        X, y = self.preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.select_model()
        self.train(X_train, y_train)
        self.evaluate(X_test, y_test)
        self.cross_validate_model(X, y)

        return self.metrics

# Additional Utility Functions

def load_data_from_folder(folder_path):
    data_files = {}
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    for csv_file in csv_files:
        csv_file_path = os.path.join(folder_path, csv_file)
        data_files[csv_file] = pd.read_csv(csv_file_path)
    return data_files

def preprocess_lstm_data(data, num_time_steps):
    sequences, targets = [], []
    for i in range(len(data) - num_time_steps):
        sequences.append(data[i:i+num_time_steps])
        targets.append(data[i+num_time_steps])
    X = np.array(sequences).reshape(-1, num_time_steps, 1)
    y = np.array(targets)
    return X, y

def train_lstm_model(X_train, y_train, lstm_units=50):
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(lstm_units),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    return model

def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    print(config.sections())
    
    # Load and preprocess data
    data_folder = config['Paths']['data_folder']
    data_files = load_data_from_folder(data_folder)

    for csv_file, data in data_files.items():
        # Preprocess data
        data = preprocess_data(data)
        target_column = 'close' if 'close' in data.columns else None
        if not target_column:
            logging.error(f"Target column not found in {csv_file}")
            continue

        X = data.drop(columns=[target_column])
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train LSTM model if applicable
        y_train = y_train.reset_index(drop=True)
        X_train_lstm, y_train_lstm = preprocess_lstm_data(y_train, 10)
        model = train_lstm_model(X_train_lstm, y_train_lstm)

if __name__ == "__main__":
    main()
