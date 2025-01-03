# src/training_app/utils.py

import matplotlib.pyplot as plt
import numpy as np
import logging
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """
        Initializes the PlotCanvas with a matplotlib Figure.
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot_confusion_matrix(self, cm, classes, title="Confusion Matrix", cmap=plt.cm.Blues):
        """
        Plots the confusion matrix on the canvas.

        Args:
            cm (array-like): Confusion matrix.
            classes (list): List of class names.
            title (str): Title of the plot.
            cmap: Colormap for the plot.
        """
        try:
            self.axes.clear()
            im = self.axes.imshow(cm, interpolation='nearest', cmap=cmap)
            self.axes.set_title(title)
            self.axes.set_xlabel("Predicted Label")
            self.axes.set_ylabel("True Label")
            tick_marks = np.arange(len(classes))
            self.axes.set_xticks(tick_marks)
            self.axes.set_xticklabels(classes, rotation=45)
            self.axes.set_yticks(tick_marks)
            self.axes.set_yticklabels(classes)

            # Loop over data dimensions and create text annotations.
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    self.axes.text(j, i, format(cm[i, j], 'd'),
                                  ha="center", va="center",
                                  color="white" if cm[i, j] > thresh else "black")

            self.fig.tight_layout()
            self.axes.figure.colorbar(im, ax=self.axes)
            self.draw()
            logging.info("Confusion matrix plotted successfully.")
        except Exception as e:
            logging.exception("Failed to plot confusion matrix.")
