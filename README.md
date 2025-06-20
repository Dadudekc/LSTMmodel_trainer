# LSTM Model Trainer

![Python version](https://img.shields.io/badge/python-3.8%2B-blue.svg) ![License](https://img.shields.io/badge/license-MIT-green.svg)

## Project Purpose

This repository showcases a lightweight machine‑learning playground built with PyQt. It demonstrates how a GUI can orchestrate dataset loading, preprocessing and model training in a single desktop app. A companion script illustrates a minimal LSTM workflow for time‑series data.

## Key Features

- **Interactive GUI** – load CSV/Excel data and kick off training from the desktop.
- **ModelTrainer class** – wraps preprocessing, model selection and evaluation.
- **Supported models** – linear regression, random forest, SVM and a small LSTM example.
- **Background training** – UI stays responsive using a separate thread.
- **Metrics & saving** – view evaluation results and optionally persist the trained model.

## Architecture Overview

```
LSTMmodel_trainer/
├── src/
│   ├── main.py            # PyQt interface and TrainingThread
│   ├── model_trainer.py   # Core training utilities
│   └── utils.py           # Plotting helper for confusion matrices
├── LSTM_Model_Trainer     # Stand‑alone script for LSTM experiments
├── tests/                 # Pytest suites using lightweight stubs
│   ├── test_imports.py
│   └── test_model_trainer_core.py
```

`model_trainer.py` can be used independently of the GUI. It handles loading a dataset, preprocessing (dummy‑encoding categorical features and dropping missing values), selecting a scikit‑learn model and computing metrics including simple cross‑validation. The GUI in `main.py` wires this trainer into a desktop application.

## Setup

Install the required packages (PyQt5, pandas, scikit‑learn, tensorflow, joblib). Example:

```bash
pip install PyQt5 pandas scikit-learn tensorflow joblib
```

## Usage

Launch the GUI:

```bash
python src/main.py
```

Run the standalone LSTM script:

```bash
python LSTM_Model_Trainer
```

Both require the dependencies above. A sample `config.ini` shows expected dataset paths.

### Running Tests

Execute the test suite with [pytest](https://docs.pytest.org/):

```bash
pytest -q
```

The tests use stub modules so they run without heavy ML dependencies.

## License

This project is released under the MIT License. See `LICENSE` for details.

