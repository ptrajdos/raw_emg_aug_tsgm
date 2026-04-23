import unittest

from sklearn.exceptions import NotFittedError

import numpy as np

from raw_emg_aug_tsgm.estimators.nn_model_estimator import NNModelEstimator
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_creators.raw_signals_creator_sines import (
    RawSignalsCreatorSines,
)


class NNModelEstimatorTest(unittest.TestCase):

    def get_models(self) -> dict:
        return {
            "Default": NNModelEstimator(),
        }

    def gen_data(self, N=10, T=50, C=2, dtype=np.float32, classes=[1, 2, 3]):
        generator = RawSignalsCreatorSines(
            set_size=N,
            samples_number=T,
            column_number=C,
            dtype=dtype,
            class_indices=classes,
        )
        return generator.get_set()

    def test_fit_predict(self):
        x_dtypes = [np.float32, np.float64]
        y_dtypes = [np.int32, np.int64, np.str_]
        classes = np.asanyarray([1, 2, 3])
        for x_dtype in x_dtypes:
            for y_dtype in y_dtypes:
                raw_data = self.gen_data(dtype=x_dtype, classes=classes.astype(y_dtype))
                for model_name, model in self.get_models().items():

                    with self.subTest(model_name=model_name, dtype=x_dtype):
                        predictions = model.fit_predict(raw_data)
                        n_objects = len(raw_data)
                        class_names = raw_data.get_labels()
                        unique_classes = np.unique(class_names)
                        self.assertIsNotNone(predictions, "Predictions are None")
                        self.assertEqual(
                            len(predictions),
                            n_objects,
                            "Number of predictions does not match number of objects",
                        )
                        self.assertTrue(
                            all(pred in unique_classes for pred in predictions),
                            "Predicted class not in original classes",
                        )

    def test_fit_predict_proba(self):
        x_dtypes = [np.float32, np.float64]
        y_dtypes = [np.int32, np.int64, np.str_]
        classes = np.asanyarray([1, 2, 3])
        for x_dtype in x_dtypes:
            for y_dtype in y_dtypes:
                raw_data = self.gen_data(dtype=x_dtype, classes=classes.astype(y_dtype))
                for model_name, model in self.get_models().items():

                    with self.subTest(model_name=model_name, dtype=x_dtype):
                        model.fit(raw_data)
                        probas = model.predict_proba(raw_data)
                        n_objects = len(raw_data)
                        class_names = raw_data.get_labels()
                        unique_classes = np.unique(class_names)
                        self.assertIsNotNone(probas, "Predictions are None")
                        self.assertIsInstance(probas, np.ndarray, "Predictions are not numpy array")
                        self.assertFalse(np.any(np.isnan(probas)), "Predictions contain NaN values")
                        self.assertFalse(np.any(np.isinf(probas)), "Predictions contain Inf values")
                        self.assertTrue(
                            np.all((probas >= 0) & (probas <= 1)),
                            "Predicted probabilities are not in [0, 1]",
                        )
                        self.assertTrue(
                            np.all(np.isclose(np.sum(probas, axis=1), 1)),
                            "Predicted probabilities do not sum to 1",
                        )
                        self.assertEqual(
                            len(probas),
                            n_objects,
                            "Number of predictions does not match number of objects",
                        )
                        self.assertEqual(
                            probas.shape[1],
                            len(unique_classes),
                            "Number of predicted classes does not match number of unique classes",
                        )
    def test_predict_before_fit(self):
        raw_data = self.gen_data()
        for model_name, model in self.get_models().items():

            with self.subTest(model_name=model_name):
                with self.assertRaises(NotFittedError, msg="Predicting before fitting should raise an exception"):
                    model.predict(raw_data)
                        

