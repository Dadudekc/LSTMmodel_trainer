# src/training_app/gui.py

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QLabel, QTableWidget, QTableWidgetItem, QComboBox,
    QHBoxLayout, QTextEdit, QProgressBar, QMessageBox, QSpinBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import pandas as pd
import logging

from training_app.model_trainer import ModelTrainer
from training_app.utils import PlotCanvas  # Import PlotCanvas for visualization

class TrainingThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)

    def __init__(self, dataset_path, target_column, model_type, hyperparameters):
        super().__init__()
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.model_type = model_type
        self.hyperparameters = hyperparameters
        self.trainer = None  # To store the trainer instance for saving the model

    def run(self):
        try:
            self.progress.emit("Loading data...")
            self.trainer = ModelTrainer(
                dataset_path=self.dataset_path,
                target_column=self.target_column,
                model_type=self.model_type,
                hyperparameters=self.hyperparameters
            )
            metrics = self.trainer.run()
            self.progress.emit("Training completed.")
            self.finished.emit(metrics)
        except Exception as e:
            logging.exception("Error during training thread execution.")
            self.progress.emit(f"Error: {str(e)}")
            self.finished.emit({})

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt Model Trainer")
        self.setGeometry(100, 100, 1000, 800)  # Increased width for better layout

        self.dataset_path = None
        self.df = None
        self.trainer = None

        self.init_ui()

    def init_ui(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        # Load Data Button
        load_data_btn = QPushButton("Load Dataset")
        load_data_btn.clicked.connect(self.load_dataset)
        layout.addWidget(load_data_btn)

        # Data Preview Table
        self.table = QTableWidget()
        layout.addWidget(self.table)

        # Model Selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Select Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Linear Regression",
            "Random Forest Regressor",
            "Random Forest Classifier",
            "SVM Regressor",
            "SVM Classifier"
        ])
        self.model_combo.currentTextChanged.connect(self.update_hyperparameter_fields)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)

        # Hyperparameter Inputs
        self.hyperparam_layout = QHBoxLayout()
        layout.addLayout(self.hyperparam_layout)
        self.update_hyperparameter_fields(self.model_combo.currentText())

        # Target Column Selection
        target_layout = QHBoxLayout()
        target_label = QLabel("Target Column:")
        self.target_combo = QComboBox()  # Changed from QLineEdit to QComboBox
        target_layout.addWidget(target_label)
        target_layout.addWidget(self.target_combo)
        layout.addLayout(target_layout)

        # Start Training Button
        self.train_btn = QPushButton("Start Training")
        self.train_btn.clicked.connect(self.start_training)
        self.train_btn.setEnabled(False)
        layout.addWidget(self.train_btn)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Training Logs
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        # Evaluation Metrics
        self.metrics_label = QLabel("Evaluation Metrics:")
        layout.addWidget(self.metrics_label)
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        layout.addWidget(self.metrics_text)

        # Plot Canvas for Confusion Matrix
        self.plot_canvas = PlotCanvas(self, width=5, height=4)
        layout.addWidget(self.plot_canvas)

        central_widget.setLayout(layout)

    def load_dataset(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Dataset",
            "",
            "CSV Files (*.csv);;Excel Files (*.xlsx *.xls)",
            options=options
        )
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self.df = pd.read_csv(file_path)
                else:
                    self.df = pd.read_excel(file_path)
                self.dataset_path = file_path
                self.display_data()
                self.populate_target_columns()
                self.train_btn.setEnabled(True)
                QMessageBox.information(self, "Success", "Dataset loaded successfully.")
                logging.info(f"Dataset loaded from {file_path}.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load dataset: {str(e)}")
                logging.exception("Failed to load dataset.")

    def display_data(self):
        if self.df is not None:
            self.table.setRowCount(0)
            self.table.setColumnCount(len(self.df.columns))
            self.table.setHorizontalHeaderLabels(self.df.columns)
            for i in range(min(len(self.df), 100)):  # Display first 100 rows
                self.table.insertRow(i)
                for j, col in enumerate(self.df.columns):
                    self.table.setItem(i, j, QTableWidgetItem(str(self.df.iloc[i, j])))
            self.table.resizeColumnsToContents()
            logging.info("Data displayed in the table.")

    def populate_target_columns(self):
        """
        Populates the target column ComboBox with column names from the dataset.
        """
        self.target_combo.clear()
        if self.df is not None:
            self.target_combo.addItems(self.df.columns)
            logging.info("Target column ComboBox populated.")

    def update_hyperparameter_fields(self, model_name):
        # Clear existing hyperparameter fields
        while self.hyperparam_layout.count():
            child = self.hyperparam_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Add hyperparameter fields based on model
        if model_name in ["Random Forest Regressor", "Random Forest Classifier"]:
            # Example hyperparameters
            n_estimators_label = QLabel("n_estimators:")
            self.n_estimators_input = QSpinBox()
            self.n_estimators_input.setRange(1, 1000)
            self.n_estimators_input.setValue(100)

            max_depth_label = QLabel("max_depth:")
            self.max_depth_input = QSpinBox()
            self.max_depth_input.setRange(1, 100)
            self.max_depth_input.setValue(10)

            self.hyperparam_layout.addWidget(n_estimators_label)
            self.hyperparam_layout.addWidget(self.n_estimators_input)
            self.hyperparam_layout.addWidget(max_depth_label)
            self.hyperparam_layout.addWidget(self.max_depth_input)
        elif model_name in ["SVM Regressor", "SVM Classifier"]:
            c_label = QLabel("C:")
            self.c_input = QDoubleSpinBox()
            self.c_input.setRange(0.01, 1000.0)
            self.c_input.setSingleStep(0.1)
            self.c_input.setValue(1.0)

            kernel_label = QLabel("Kernel:")
            self.kernel_combo = QComboBox()
            self.kernel_combo.addItems(["linear", "poly", "rbf", "sigmoid"])

            self.hyperparam_layout.addWidget(c_label)
            self.hyperparam_layout.addWidget(self.c_input)
            self.hyperparam_layout.addWidget(kernel_label)
            self.hyperparam_layout.addWidget(self.kernel_combo)
        else:
            # Linear Regression has no hyperparameters
            info_label = QLabel("No hyperparameters to configure for Linear Regression.")
            self.hyperparam_layout.addWidget(info_label)
            logging.info("Linear Regression selected. No hyperparameters to configure.")

    def start_training(self):
        if not self.dataset_path:
            QMessageBox.warning(self, "Warning", "Please load a dataset first.")
            return

        target_column = self.target_combo.currentText().strip()
        if not target_column:
            QMessageBox.warning(self, "Warning", "Please select the target column.")
            return

        if target_column not in self.df.columns:
            QMessageBox.warning(self, "Warning", f"'{target_column}' is not a valid column in the dataset.")
            return

        model_name = self.model_combo.currentText()
        model_type = self.map_model_name_to_type(model_name)

        # Gather hyperparameters
        hyperparameters = {}
        if model_type in ['random_forest_regressor', 'random_forest_classifier']:
            hyperparameters['n_estimators'] = self.n_estimators_input.value()
            hyperparameters['max_depth'] = self.max_depth_input.value()
        elif model_type in ['svm_regressor', 'svm_classifier']:
            hyperparameters['C'] = self.c_input.value()
            hyperparameters['kernel'] = self.kernel_combo.currentText()

        # Disable training button to prevent multiple clicks
        self.train_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        self.metrics_text.clear()
        self.plot_canvas.axes.clear()
        self.plot_canvas.draw()

        logging.info("Starting training process.")

        # Start training in a separate thread
        self.thread = TrainingThread(
            dataset_path=self.dataset_path,
            target_column=target_column,
            model_type=model_type,
            hyperparameters=hyperparameters
        )
        self.thread.progress.connect(self.update_log)
        self.thread.finished.connect(self.training_finished)
        self.thread.start()

    def map_model_name_to_type(self, model_name):
        mapping = {
            "Linear Regression": "linear_regression",
            "Random Forest Regressor": "random_forest_regressor",
            "Random Forest Classifier": "random_forest_classifier",
            "SVM Regressor": "svm_regressor",
            "SVM Classifier": "svm_classifier"
        }
        return mapping.get(model_name, "linear_regression")

    def update_log(self, message):
        self.log_text.append(message)
        if "completed" in message.lower():
            self.progress_bar.setValue(100)
            logging.info("Training completed.")
        elif "error" in message.lower():
            self.progress_bar.setValue(0)
            self.train_btn.setEnabled(True)
            logging.error(f"Training error: {message}")
        else:
            # Assuming progress is at 50% during training
            self.progress_bar.setValue(50)
            logging.info(f"Training progress: {message}")

    def training_finished(self, metrics):
        if metrics:
            self.metrics_text.setPlainText("\n".join([f"{k}: {v}" for k, v in metrics.items()]))
            QMessageBox.information(self, "Success", "Model training and evaluation completed.")
            logging.info("Model training and evaluation completed.")

            # Plot confusion matrix if classification
            if 'Confusion Matrix' in metrics and 'Classes' in metrics:
                try:
                    cm = metrics['Confusion Matrix']
                    classes = metrics['Classes']
                    self.plot_canvas.plot_confusion_matrix(cm, classes)
                    logging.info("Confusion matrix plotted.")
                except Exception as e:
                    logging.exception("Failed to plot confusion matrix.")

            # Prompt to save the model
            save_model = QMessageBox.question(
                self, "Save Model", "Do you want to save the trained model?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if save_model == QMessageBox.Yes:
                self.save_model()
        else:
            QMessageBox.critical(self, "Error", "Model training failed. Check logs for details.")
            logging.error("Model training failed.")

        self.train_btn.setEnabled(True)

    def save_model(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Model",
            "",
            "Joblib Files (*.joblib);;All Files (*)",
            options=options
        )
        if file_path:
            try:
                # Check if thread and trainer are available
                if not hasattr(self, 'thread') or not self.thread.trainer:
                    raise ValueError("No trained model available to save.")
                trainer = self.thread.trainer
                trainer.save_model(file_path)
                QMessageBox.information(self, "Success", f"Model saved to {file_path}")
                logging.info(f"Model saved to {file_path}.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save model: {str(e)}")
                logging.exception("Failed to save model.")
