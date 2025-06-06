# Tests for core functionality of ModelTrainer
import importlib
import sys
import types
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))


class DummyDataFrame:
    def __init__(self, rows):
        self._rows = list(rows)
        self.columns = list(rows[0].keys()) if rows else []

    def dropna(self):
        return self

    def drop(self, columns=None, **kwargs):
        if columns is None:
            return self
        if isinstance(columns, (list, tuple, set)):
            cols = set(columns)
        else:
            cols = {columns}
        new_rows = [{k: v for k, v in r.items() if k not in cols} for r in self._rows]
        return DummyDataFrame(new_rows)

    def __getitem__(self, col):
        return [r[col] for r in self._rows]

    def __len__(self):
        return len(self._rows)

    def reset_index(self, drop=False):
        return self


def setup_dummy_modules(dataset):
    pd_mod = types.ModuleType('pandas')
    pd_mod.read_csv = lambda path: dataset
    pd_mod.get_dummies = lambda df, drop_first=True: df
    sys.modules['pandas'] = pd_mod

    np_mod = types.ModuleType('numpy')
    sys.modules['numpy'] = np_mod

    class DummyModel:
        def fit(self, X, y):
            self.const = sum(y) / len(y)

        def predict(self, X):
            return [getattr(self, 'const', 0)] * len(X)

    linear_model = types.ModuleType('linear_model')
    linear_model.LinearRegression = DummyModel
    sys.modules['sklearn'] = types.ModuleType('sklearn')
    sys.modules['sklearn.linear_model'] = linear_model

    ensemble = types.ModuleType('ensemble')
    ensemble.RandomForestRegressor = DummyModel
    ensemble.RandomForestClassifier = DummyModel
    sys.modules['sklearn.ensemble'] = ensemble

    svm = types.ModuleType('svm')
    svm.SVR = DummyModel
    svm.SVC = DummyModel
    sys.modules['sklearn.svm'] = svm

    metrics = types.ModuleType('metrics')

    def mean_squared_error(y_true, y_pred):
        return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)

    def r2_score(y_true, y_pred):
        mean_y = sum(y_true) / len(y_true)
        ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
        ss_tot = sum((yt - mean_y) ** 2 for yt in y_true)
        return 1 - ss_res / ss_tot if ss_tot else 0

    metrics.mean_squared_error = mean_squared_error
    metrics.r2_score = r2_score
    metrics.accuracy_score = lambda *a, **k: 1.0
    metrics.precision_score = lambda *a, **k: 1.0
    metrics.recall_score = lambda *a, **k: 1.0
    metrics.f1_score = lambda *a, **k: 1.0
    sys.modules['sklearn.metrics'] = metrics

    model_selection = types.ModuleType('model_selection')

    def train_test_split(X, y, test_size=0.2, random_state=None):
        split = int(len(X) * (1 - test_size))
        if hasattr(X, '_rows'):
            x_train = DummyDataFrame(X._rows[:split])
            x_test = DummyDataFrame(X._rows[split:])
        else:
            x_train, x_test = X[:split], X[split:]
        return x_train, x_test, y[:split], y[split:]

    class ScoreList(list):
        def mean(self):
            return sum(self) / len(self)

        def std(self):
            m = self.mean()
            return (sum((x - m) ** 2 for x in self) / len(self)) ** 0.5

    model_selection.train_test_split = train_test_split
    model_selection.cross_val_score = lambda *a, cv=5, **k: ScoreList([1.0] * cv)
    sys.modules['sklearn.model_selection'] = model_selection

    preprocessing = types.ModuleType('preprocessing')
    for scaler in ['StandardScaler', 'PowerTransformer', 'MinMaxScaler', 'RobustScaler',
                   'QuantileTransformer', 'Normalizer', 'MaxAbsScaler']:
        setattr(preprocessing, scaler, object)
    sys.modules['sklearn.preprocessing'] = preprocessing

    joblib_mod = types.ModuleType('joblib')
    joblib_mod.dump = lambda model, path: open(path, 'w').write('data')
    joblib_mod.load = lambda path: DummyModel()
    sys.modules['joblib'] = joblib_mod

    tf_mod = types.ModuleType('tensorflow')
    keras_mod = types.ModuleType('keras')
    sys.modules['tensorflow'] = tf_mod
    sys.modules['tensorflow.keras'] = keras_mod
    models_mod = types.ModuleType('models')
    models_mod.Sequential = object
    sys.modules['tensorflow.keras.models'] = models_mod
    layers_mod = types.ModuleType('layers')
    layers_mod.Dense = object
    layers_mod.LSTM = object
    sys.modules['tensorflow.keras.layers'] = layers_mod

    kt_mod = types.ModuleType('keras_tuner')
    kt_mod.HyperModel = object
    kt_mod.RandomSearch = object
    sys.modules['keras_tuner'] = kt_mod



def test_run_linear_regression(tmp_path):
    dataset = DummyDataFrame([
        {'feat': i, 'target': i * 2} for i in range(10)
    ])
    setup_dummy_modules(dataset)
    module = importlib.import_module('model_trainer')
    importlib.reload(module)
    trainer = module.ModelTrainer('dummy.csv', 'target', 'linear_regression', {})
    metrics = trainer.run()
    assert 'Mean Squared Error' in metrics
    assert 'R² Score' in metrics
    assert 'Cross-Validation R² Mean' in metrics


