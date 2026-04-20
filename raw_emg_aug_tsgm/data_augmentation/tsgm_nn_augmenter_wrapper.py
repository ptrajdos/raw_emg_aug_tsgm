from typing import Optional, Any
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter import (
    RawSignalsAugumenter,
)
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.sampler_mixin import (
    SamplerMixin,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
import numpy as np
from sklearn.dummy import check_random_state
from sklearn.exceptions import NotFittedError

from copy import deepcopy

from tsgm.models.architectures.zoo import Architecture
from tsgm.models.cgan import ConditionalGAN

from raw_emg_aug_tsgm.data_augmentation.model_factories.model_factory import ModelFactory
from raw_emg_aug_tsgm.raw_signals.tools import convert_to_tensor


class TSGMANNAugmenterWrapper(SamplerMixin, RawSignalsAugumenter):

    def __init__(
        self,
        model_factory:ModelFactory,
        fit_options: Optional[dict] = None,
        random_state=10,
    ) -> None:
        RawSignalsAugumenter.__init__(self)
        SamplerMixin.__init__(self)

        self.model_factory = model_factory
        self.fit_options = fit_options
        self.random_state = random_state


    def _get_effective_fit_options(self):
        return {} if self.fit_options is None else self.fit_options

    def fit(
        self,
        raw_signals: RawSignals,
        **kwargs
    ) -> RawSignalsAugumenter:
        self._is_fitted = True

        self._random_state = check_random_state(self.random_state)

        self._model = self.model_factory.get_compiled_model(raw_signals=raw_signals)
        fit_args = deepcopy(kwargs)
        fit_args.update(self._get_effective_fit_options())

        X = convert_to_tensor(raw_signals)
        y = raw_signals.get_labels().reshape(-1,1)

        self.classes_, counts = np.unique(y, return_counts=True)
        self.probs_ = counts/np.sum(counts)

        self._model.fit(X,y, **fit_args)

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
        labels_to_gen = self._random_state.choice(self.classes_, size=n_gen_sigs, p=self.probs_)
        tensor = self._model.generate(labels_to_gen.reshape(-1,1)).numpy().astype(rs_dtype)

        for np_sig, label in  zip(tensor, labels_to_gen):
            new_sig:RawSignal = deepcopy(tmp_signal)
            new_sig.signal = np_sig
            new_sig.set_label(label)
            new_signals.append(new_sig)

        return new_signals

    def fit_transform(self, raw_signals: RawSignals) -> RawSignals:
        return self.fit(raw_signals).transform(raw_signals)
