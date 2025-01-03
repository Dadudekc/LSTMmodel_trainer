# src/utils.py

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

    def plot_confusion_matrix(self, cm, classes):
        self.axes.clear()
        self.axes.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        self.axes.set_title("Confusion Matrix")
        self.axes.set_xlabel("Predicted Label")
        self.axes.set_ylabel("True Label")
        tick_marks = np.arange(len(classes))
        self.axes.set_xticks(tick_marks)
        self.axes.set_xticklabels(classes, rotation=45)
        self.axes.set_yticks(tick_marks)
        self.axes.set_yticklabels(classes)
        for i in range(len(classes)):
            for j in range(len(classes)):
                self.axes.text(j, i, cm[i, j],
                               horizontalalignment="center",
                               color="white" if cm[i, j] > cm.max()/2 else "black")
        self.axes.figure.tight_layout()
        self.draw()
