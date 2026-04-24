from copy import deepcopy
from typing import Optional

import keras
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.exceptions import NotFittedError

from raw_emg_aug_tsgm.estimators.model_factories.classification_model_factory import (
    ClassificationModelFactory,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals


class NNModelEstimator:

    def __init__(
        self,
        model_factory: Optional[ClassificationModelFactory] = None,
        forced_dtype=np.float32,
        fit_options: Optional[dict] = None,
        predict_options: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.model_factory = model_factory
        self.forced_dtype = forced_dtype
        self.fit_options = fit_options
        self.predict_options = predict_options

    def _get_effective_model_factory(self) -> ClassificationModelFactory:
        return (
            ClassificationModelFactory()
            if self.model_factory is None
            else self.model_factory
        )  # type: ignore
    
    def _get_effective_fit_options(self):
        return {} if self.fit_options is None else self.fit_options

    def _get_effective_predict_options(self):
        return {} if self.predict_options is None else self.predict_options

    def fit(self, raw_signals: RawSignals, y=None, **fit_params):

        effective_factory = self._get_effective_model_factory()
        self.model_ = effective_factory.get_compiled_model(raw_signals)

        np_X = raw_signals.to_numpy().astype(self.forced_dtype)
        labels = raw_signals.get_labels()
        self.classes_ = np.unique(labels)
        self.label_encoder = LabelEncoder()
        labels_enc = self.label_encoder.fit_transform(labels)
        labels_enc = np.asanyarray(labels_enc).reshape(-1, 1).astype(self.forced_dtype)
        lebels_enc_keras = keras.utils.to_categorical(
            labels_enc, num_classes=len(self.classes_)
        )

        fit_params_e = deepcopy(fit_params)
        fit_params_e.update(self._get_effective_fit_options())
        self.model_.fit(np_X, lebels_enc_keras, **fit_params_e)

        self._is_fitted = True

        return self

    def fit_predict(self, raw_signals: RawSignals, y=None, **fit_params):
        self.fit(raw_signals, y=y, **fit_params)
        return self.predict(raw_signals)

    def _check_fitted(self):
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise NotFittedError("Model is not fitted yet")

    def predict_proba(self, raw_signals: RawSignals, **predict_params):
        self._check_fitted()
        predict_params_e = deepcopy(predict_params)
        predict_params_e.update(self._get_effective_predict_options())
        X_np = raw_signals.to_numpy().astype(self.forced_dtype)
        return self.model_.predict(X_np, **predict_params_e)

    def predict(self, raw_signals: RawSignals, **predict_params):
        proba = self.predict_proba(raw_signals, **predict_params)
        class_indices = np.argmax(proba, axis=1)
        return self.label_encoder.inverse_transform(class_indices)
