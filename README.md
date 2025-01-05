# LSTM Model Trainer

A PyQt-based application for training machine learning models, including LSTM networks, with a user-friendly interface for data preprocessing, model configuration, and performance evaluation.



## Project Directory Structure

The following outlines the directory structure of the project and its purpose:
# Project Directory Structure

The following outlines the directory structure of the project and its purpose:

```plaintext
LSTMModelTrainer/
├── data/                # Primary folder for datasets used in model training.
├── data2/               # Secondary folder for additional datasets.
├── models/              # Folder to save trained models.
├── plots/               # Folder for storing visualization plots.
├── src/                 # Source code for the project.
│   ├── main.py          # Main PyQt GUI application.
│   ├── model_trainer.py # Core model training logic and utilities.
│   ├── utils.py         # Auxiliary utilities for data processing and configuration.
├── .gitignore           # Excludes unnecessary files from version control.
├── config.ini           # Configuration file for setting default paths and parameters.
├── LICENSE              # License information for the project.
├── LSTM_Model_Trainer/  # Repository root folder (includes documentation and other assets).
└── README.md            # Project overview and documentation.


## Current Features

- **Interactive GUI**: A user-friendly interface for loading datasets, configuring models, and monitoring progress.
- **Data Handling**:
  - Support for CSV and Excel files.
  - Preprocessing steps like handling missing values, scaling, and encoding categorical variables.
- **Model Support**:
  - Linear Regression, Random Forest (Classifier & Regressor), SVM (Classifier & Regressor), and LSTM.
- **Training and Evaluation**:
  - Real-time logs and progress updates.
  - Cross-validation and metric evaluation.
  - Save and load models for reuse.

---

## Future Plans

### Short-Term Goals
- Expand preprocessing options to support additional feature engineering techniques.
- Enhance LSTM functionality with more hyperparameter tuning options.
- Implement real-time visual feedback for metrics like loss and accuracy during training.

### Medium-Term Goals
- Add support for additional machine learning models:
  - Gradient Boosting (e.g., XGBoost, LightGBM).
  - Neural Networks for complex and deeper learning tasks.
- Enable support for diverse file formats, such as JSON and SQL-based datasets.
- Refine the GUI for an improved and intuitive user experience.

### Long-Term Goals
- **Cloud Integration**:
  - Load datasets directly from cloud storage services.
  - Deploy trained models to cloud-based platforms for real-time inference.
- **Distributed Training**:
  - Introduce capabilities for training across multiple GPUs or distributed systems.
- Build an integrated tutorial system within the application to guide and onboard new users effectively.

---

## Installation and Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Dadudekc/LSTM_model_trainer.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd LSTMModelTrainer
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python src/main.py
   ```

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to suggest features or improvements.

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.

---

## Contact

- **Developer**: Dadudekc  
- **Email**: Freerideinvestor@gmail.com
- **Discord**: Dadudekc#1234  
- **Repository**: [LSTM Model Trainer](https://github.com/Dadudekc/LSTM_model_trainer)

---
