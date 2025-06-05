import importlib
import sys
import types
from pathlib import Path

# Ensure src directory is on the path so imports work
SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))


def test_import_model_trainer():
    dummy = types.ModuleType('dummy')
    sys.modules.setdefault('numpy', dummy)
    sys.modules.setdefault('pandas', dummy)
    sys.modules.setdefault('sklearn', types.ModuleType('sklearn'))
    linear_model = types.ModuleType('linear_model')
    setattr(linear_model, 'LinearRegression', object)
    sys.modules.setdefault('sklearn.linear_model', linear_model)
    ensemble = types.ModuleType('ensemble')
    setattr(ensemble, 'RandomForestRegressor', object)
    setattr(ensemble, 'RandomForestClassifier', object)
    sys.modules.setdefault('sklearn.ensemble', ensemble)
    svm = types.ModuleType('svm')
    setattr(svm, 'SVR', object)
    setattr(svm, 'SVC', object)
    sys.modules.setdefault('sklearn.svm', svm)
    metrics = types.ModuleType('metrics')
    for name in ['mean_squared_error', 'r2_score', 'accuracy_score', 'precision_score', 'recall_score', 'f1_score']:
        setattr(metrics, name, lambda *a, **k: 0)
    sys.modules.setdefault('sklearn.metrics', metrics)
    model_selection = types.ModuleType('model_selection')
    setattr(model_selection, 'train_test_split', lambda X, y, test_size=0.2, random_state=None: (X, X, y, y))
    setattr(model_selection, 'cross_val_score', lambda *a, **k: [0])
    sys.modules.setdefault('sklearn.model_selection', model_selection)
    preprocessing = types.ModuleType('preprocessing')
    for scaler in ['StandardScaler', 'PowerTransformer', 'MinMaxScaler', 'RobustScaler', 'QuantileTransformer', 'Normalizer', 'MaxAbsScaler']:
        setattr(preprocessing, scaler, object)
    sys.modules.setdefault('sklearn.preprocessing', preprocessing)
    sys.modules.setdefault('tensorflow', types.ModuleType('tensorflow'))
    sys.modules.setdefault('tensorflow.keras', types.ModuleType('keras'))
    keras_models = types.ModuleType('models')
    setattr(keras_models, 'Sequential', object)
    sys.modules.setdefault('tensorflow.keras.models', keras_models)
    keras_layers = types.ModuleType('layers')
    setattr(keras_layers, 'Dense', object)
    setattr(keras_layers, 'LSTM', object)
    sys.modules.setdefault('tensorflow.keras.layers', keras_layers)
    kt = types.ModuleType('keras_tuner')
    setattr(kt, 'HyperModel', object)
    setattr(kt, 'RandomSearch', object)
    sys.modules.setdefault('keras_tuner', kt)
    sys.modules.setdefault('joblib', types.ModuleType('joblib'))
    module = importlib.import_module('model_trainer')
    assert hasattr(module, 'ModelTrainer')


def test_import_main():
    dummy = types.ModuleType('dummy')
    sys.modules.setdefault('PyQt5', dummy)
    widgets = types.ModuleType('widgets')
    core = types.ModuleType('core')
    for attr in [
        'QApplication', 'QMainWindow', 'QWidget', 'QVBoxLayout', 'QPushButton',
        'QFileDialog', 'QLabel', 'QTableWidget', 'QTableWidgetItem', 'QComboBox',
        'QLineEdit', 'QHBoxLayout', 'QTextEdit', 'QProgressBar', 'QMessageBox',
        'QSpinBox', 'QDoubleSpinBox'
    ]:
        setattr(widgets, attr, object)
    setattr(core, 'Qt', object)
    setattr(core, 'QThread', object)
    setattr(core, 'pyqtSignal', lambda *a, **k: None)
    sys.modules.setdefault('PyQt5.QtWidgets', widgets)
    sys.modules.setdefault('PyQt5.QtCore', core)
    module = importlib.import_module('main')
    assert hasattr(module, 'MainWindow')
