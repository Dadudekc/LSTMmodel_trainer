# src/tests/test_model_trainer.py

import unittest
import os
import pandas as pd
from training_app.model_trainer import ModelTrainer

class TestModelTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up a sample dataset for testing.
        """
        cls.test_csv = 'src/tests/data/sample.csv'
        os.makedirs(os.path.dirname(cls.test_csv), exist_ok=True)
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'target': [1, 0, 1, 0, 1]
        })
        df.to_csv(cls.test_csv, index=False)

    @classmethod
    def tearDownClass(cls):
        """
        Clean up the sample dataset after tests.
        """
        if os.path.exists(cls.test_csv):
            os.remove(cls.test_csv)

    def setUp(self):
        """
        Initialize ModelTrainer instance before each test.
        """
        self.trainer = ModelTrainer(
            dataset_path=self.test_csv,
            target_column='target',
            model_type='random_forest_classifier',
            hyperparameters={'n_estimators': 10, 'max_depth': 5}
        )

    def test_load_data(self):
        """
        Test if data is loaded correctly.
        """
        data = self.trainer.load_data()
        self.assertIsNotNone(data)
        self.assertIn('target', data.columns)
        self.assertEqual(len(data), 5)

    def test_preprocess_data(self):
        """
        Test data preprocessing steps.
        """
        data = self.trainer.load_data()
        X, y = self.trainer.preprocess_data(data)
        self.assertEqual(X.shape[0], y.shape[0])
        self.assertEqual(X.shape[1], 2)  # Two features after encoding
        self.assertTrue((X != 0).all())  # Assuming no scaling results in zeros

    def test_select_model(self):
        """
        Test model selection based on model_type.
        """
        self.trainer.select_model()
        self.assertIsNotNone(self.trainer.model)
        self.assertEqual(self.trainer.model.n_estimators, 10)
        self.assertEqual(self.trainer.model.max_depth, 5)

    def test_train(self):
        """
        Test the training process.
        """
        data = self.trainer.load_data()
        X, y = self.trainer.preprocess_data(data)
        self.trainer.X_train, self.trainer.X_test, self.trainer.y_train, self.trainer.y_test = self.trainer.run().get('X_train'), self.trainer.run().get('X_test'), self.trainer.run().get('y_train'), self.trainer.run().get('y_test')
        self.trainer.select_model()
        self.trainer.train(X[:4], y[:4])  # Train on first 4 samples
        self.assertIsNotNone(self.trainer.model)
        self.assertTrue(hasattr(self.trainer.model, 'predict'))

    def test_evaluate_regression(self):
        """
        Test evaluation metrics for regression models.
        """
        # Initialize a regression model
        regression_trainer = ModelTrainer(
            dataset_path=self.test_csv,
            target_column='target',
            model_type='linear_regression',
            hyperparameters={}
        )
        data = regression_trainer.load_data()
        X, y = regression_trainer.preprocess_data(data)
        X_train, X_test, y_train, y_test = regression_trainer.run()
        regression_trainer.select_model()
        regression_trainer.train(X_train, y_train)
        regression_trainer.evaluate(X_test, y_test)
        metrics = regression_trainer.metrics
        self.assertIn('Mean Squared Error', metrics)
        self.assertIn('R² Score', metrics)

    def test_evaluate_classification(self):
        """
        Test evaluation metrics for classification models.
        """
        self.trainer.run()
        metrics = self.trainer.metrics
        self.assertIn('Accuracy', metrics)
        self.assertIn('Precision', metrics)
        self.assertIn('Recall', metrics)
        self.assertIn('F1 Score', metrics)
        self.assertIn('Confusion Matrix', metrics)
        self.assertIn('Classes', metrics)

    def test_cross_validate_regression(self):
        """
        Test cross-validation for regression models.
        """
        regression_trainer = ModelTrainer(
            dataset_path=self.test_csv,
            target_column='target',
            model_type='linear_regression',
            hyperparameters={}
        )
        regression_trainer.run()
        regression_trainer.cross_validate_model(regression_trainer.X_train, regression_trainer.y_train)
        metrics = regression_trainer.metrics
        self.assertIn('Cross-Validation R² Mean', metrics)
        self.assertIn('Cross-Validation R² Std', metrics)

    def test_cross_validate_classification(self):
        """
        Test cross-validation for classification models.
        """
        self.trainer.run()
        self.trainer.cross_validate_model(self.trainer.X_train, self.trainer.y_train)
        metrics = self.trainer.metrics
        self.assertIn('Cross-Validation Accuracy Mean', metrics)
        self.assertIn('Cross-Validation Accuracy Std', metrics)

    def test_save_and_load_model(self):
        """
        Test saving and loading the trained model.
        """
        self.trainer.run()
        save_path = 'src/tests/data/test_model.joblib'
        self.trainer.save_model(save_path)
        self.assertTrue(os.path.exists(save_path))

        # Load the model
        loaded_trainer = ModelTrainer(
            dataset_path=self.test_csv,
            target_column='target',
            model_type='random_forest_classifier',
            hyperparameters={'n_estimators': 10, 'max_depth': 5}
        )
        loaded_trainer.load_model(save_path)
        self.assertIsNotNone(loaded_trainer.model)
        os.remove(save_path)  # Clean up

if __name__ == '__main__':
    unittest.main()
