# LSTM Model Trainer

A small PyQt-based application for experimenting with machine-learning models. The goal is to showcase how automation and ML tooling can be orchestrated in Python.

![Python version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

The GUI lets you load a CSV or Excel file, select a model and launch training in a background thread.  `ModelTrainer` handles preprocessing, training and evaluation before returning metrics that can be displayed and saved.  A standalone script (`LSTM_Model_Trainer`) demonstrates training an LSTM on time-series data.

```
GUI (PyQt)
   │
   ├── Load dataset
   ├── Select model & hyperparameters
   ├── Launch TrainingThread
   │       └── ModelTrainer
   │              ├── preprocess data
   │              ├── train model
   │              └── evaluate & cross‑validate
   └── Display metrics / save model
```

Supported models include linear regression, random forests, SVMs and a simple LSTM example. The tests under `tests/` confirm imports and core training logic.

## Project Structure

```plaintext
LSTMmodel_trainer/
├── src/
│   ├── main.py            # PyQt interface
│   ├── model_trainer.py   # Training utilities
│   └── utils.py           # Misc helpers
├── LSTM_Model_Trainer     # Standalone LSTM training script
├── tests/
│   ├── test_imports.py
│   └── test_model_trainer_core.py
├── config.ini             # Example paths for datasets
├── LICENSE
├── README.md
└── interview_summary.txt
```

## Key Features

- **Interactive GUI** for basic model training
- **Data Handling** for CSV and Excel files
- **Model Options**: linear regression, random forest, SVM and LSTM
- **Asynchronous Training** with log updates and metric display

## Installation

Install the minimal dependencies:

```bash
pip install PyQt5 pandas scikit-learn tensorflow joblib
```

## Usage

Start the GUI:

```bash
python src/main.py
```

Run the tests with `pytest`.

## License

Released under the MIT License.

## Contact

**Developer:** Dadudekc – [LSTM Model Trainer](https://github.com/Dadudekc/LSTM_model_trainer)
