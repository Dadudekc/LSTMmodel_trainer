# Codebase Overview

This document provides a quick tour of the repository to help new contributors understand where things live.

## Directory Layout

- **src/** – main application code
  - `main.py` – PyQt5 desktop interface that uses `TrainingThread` to run models without blocking the UI.
  - `model_trainer.py` – `ModelTrainer` class with data loading, preprocessing, model selection and evaluation logic. Includes helper functions for simple LSTM experiments.
  - `utils.py` – helper for plotting confusion matrices within the GUI.
- **LSTM_Model_Trainer** – stand‑alone script showing an LSTM training flow using `config.ini` for dataset paths.
- **tests/** – unit tests built with pytest. Heavy dependencies are stubbed so tests run quickly even in minimal environments.
- **config.ini** – example configuration referencing dataset folders on disk.

## Entry Points

`python src/main.py` launches the GUI.

`python LSTM_Model_Trainer` runs the console demonstration (requires the dependencies listed in the README).

## Dependencies

The project expects Python 3.8+ with `PyQt5`, `pandas`, `scikit-learn`, `tensorflow` and `joblib` installed. The test suite can run without them thanks to stub modules.

