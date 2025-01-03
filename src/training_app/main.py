# src/main.py

import sys
import logging
from PyQt5.QtWidgets import QApplication
from LSTMModelTrainer.src.training_app.gui import MainWindow

def setup_logging():
    """
    Configures logging for the application.
    Logs are saved to 'app.log' with INFO level and above.
    """
    logging.basicConfig(
        filename='app.log',
        filemode='a',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

def main():
    """
    The main entry point of the application.
    Sets up logging, initializes the QApplication, and displays the MainWindow.
    """
    try:
        setup_logging()
        logging.info("Application started.")
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        logging.exception("An unexpected error occurred.")
        sys.exit(1)

if __name__ == "__main__":
    main()
