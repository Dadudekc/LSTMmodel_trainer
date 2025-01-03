# src/tests/test_utils.py

import unittest
import numpy as np
from training_app.utils import PlotCanvas
from PyQt5.QtWidgets import QApplication
import sys

class TestPlotCanvas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up a QApplication instance required for PlotCanvas.
        """
        cls.app = QApplication(sys.argv)

    def setUp(self):
        """
        Initialize PlotCanvas before each test.
        """
        self.canvas = PlotCanvas(width=5, height=4)

    def test_plot_confusion_matrix(self):
        """
        Test plotting a confusion matrix.
        """
        cm = np.array([[5, 2], [1, 7]])
        classes = ['Class 0', 'Class 1']
        try:
            self.canvas.plot_confusion_matrix(cm, classes)
            # If no exception is raised, the test passes
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"plot_confusion_matrix raised an exception {e}")

    def tearDown(self):
        """
        Clean up after each test.
        """
        self.canvas.close()

    @classmethod
    def tearDownClass(cls):
        """
        Clean up the QApplication instance after all tests.
        """
        cls.app.quit()

if __name__ == '__main__':
    unittest.main()
