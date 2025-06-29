# Product Requirements Document (PRD)

## Overview
This project delivers a lightweight desktop application for experimenting with machine-learning models. It exposes a simple PyQt interface and a command line script that share a common `ModelTrainer` component.

## Goals
- Allow non-technical users to load a dataset (CSV or Excel) and train a variety of models with minimal setup.
- Support quick experimentation with common algorithms: linear regression, random forest, SVM, PPO reinforcement learning and a basic LSTM.
- Keep the training pipeline easy to extend and understand so contributors can add new models or preprocessing steps.

## Non-Goals
- Full production deployment or large-scale data management.
- Advanced hyperparameter tuning or deep experiment tracking.

## Functional Requirements
- Desktop GUI to load a dataset, configure a model and view results.
- Command line script `LSTM_Model_Trainer` for headless training.
- Data preprocessing: drop missing values and dummy-encode categoricals.
- Evaluation metrics displayed after training and optional cross-validation.
- Ability to save trained models to disk.
- Minimal unit tests verifying the `ModelTrainer` logic and imports.

## Non-Functional Requirements
- Python 3.8+ environment with PyQt5, pandas, scikit-learn, tensorflow and joblib installed.
- Code should run on Windows/Linux/macOS with minimal modification.
- Keep external dependencies lightweight and avoid network access during tests.

## Acceptance Criteria
- Users can train a model via the GUI without the interface freezing.
- The command line script outputs evaluation metrics in JSON format.
- Tests run successfully using `pytest -q` with stubbed dependencies.
- Documentation covers setup, usage and testing procedures.

## Status
This PRD has been reviewed and finalized for the upcoming **v0.2 beta** release.

