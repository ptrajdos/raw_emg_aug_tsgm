from copy import deepcopy
from typing import Optional

import numpy as np
from sklearn.preprocessing import LabelEncoder
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter import (
    RawSignalsAugumenter,
)
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.sampler_mixin import (
    SamplerMixin,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from sklearn.dummy import check_random_state
from sklearn.exceptions import NotFittedError

from raw_emg_aug_tsgm.data_augmentation.model_factories.model_factory import (
    ModelFactory,
)
from raw_emg_aug_tsgm.raw_signals.tools import convert_to_tensor

from tsgm.utils.data_processing import TSFeatureWiseScaler, TSGlobalScaler


class TSGMANNAugmenterWrapper(SamplerMixin, RawSignalsAugumenter):

    def __init__(
        self,
        model_factory: ModelFactory,
        fit_options: Optional[dict] = None,
        random_state=10,
        normalize: bool = True,
        normalize_channels: bool = False,
    ) -> None:
        RawSignalsAugumenter.__init__(self)
        SamplerMixin.__init__(self)

        self.model_factory = model_factory
        self.fit_options = fit_options
        self.random_state = random_state
        self.normalize = normalize
        self.normalize_channels = normalize_channels

    def _get_effective_fit_options(self):
        return {} if self.fit_options is None else self.fit_options

    def _set_effective_normalizer(self):
        if not self.normalize:
            self._normalizer = None
            return

        self._normalizer = (
            TSFeatureWiseScaler((-1, 1))
            if self.normalize_channels
            else TSGlobalScaler()
        )

    def _normalize_input(self, X):
        if self._normalizer is not None:
            Xn = self._normalizer.fit_transform(X)
            return Xn

        return X

    def _denormalize_output(self, X):
        if self._normalizer is not None:
            Xn = self._normalizer.inverse_transform(X)
            return Xn

        return X

    def fit(self, raw_signals: RawSignals, **kwargs) -> RawSignalsAugumenter:
        self._is_fitted = True

        self._random_state = check_random_state(self.random_state)

        self._model = self.model_factory.get_compiled_model(raw_signals=raw_signals)
        fit_args = deepcopy(kwargs)
        fit_args.update(self._get_effective_fit_options())

        X = convert_to_tensor(raw_signals).astype(np.float32)
        self._set_effective_normalizer()
        X = self._normalize_input(X)

        y = raw_signals.get_labels()
        self.label_encoder_ = LabelEncoder()
        enc_y = self.label_encoder_.fit_transform(y)
        y = np.asanyarray(enc_y).reshape(-1, 1)

        self.classes_, counts = np.unique(y, return_counts=True)
        self.probs_ = counts / np.sum(counts)

        self._model.fit(X, y, **fit_args)

        return self

    def _check_if_fitted(self):
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise NotFittedError(
                "You must fit the augumenter before calling transform. Call fit() or fit_transform() first."
            )

    def transform(self, raw_signals: RawSignals) -> RawSignals:
        self._check_if_fitted()

        new_signals = raw_signals.initialize_empty()
        tmp_signal = deepcopy(raw_signals[0])
        rs_dtype = raw_signals[0].to_numpy().dtype

        n_gen_sigs = len(raw_signals)
        labels_to_gen = self._random_state.choice(
            self.classes_, size=n_gen_sigs, p=self.probs_
        )
        decoded_labels = self.label_encoder_.inverse_transform(labels_to_gen)
        # TODO need retransformation!
        tensor = self._model.generate(labels_to_gen.reshape(-1, 1))
        tensor_denorm = self._denormalize_output(tensor).numpy().astype(rs_dtype)

        for np_sig, label in zip(tensor_denorm, decoded_labels):
            new_sig: RawSignal = deepcopy(tmp_signal)
            new_sig.signal = np_sig
            new_sig.set_label(label)
            new_signals.append(new_sig)

        return new_signals

    def fit_transform(self, raw_signals: RawSignals) -> RawSignals:
        return self.fit(raw_signals).transform(raw_signals)
