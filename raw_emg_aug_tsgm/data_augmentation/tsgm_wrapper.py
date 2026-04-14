from typing import Optional
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter import (
    RawSignalsAugumenter,
)
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.sampler_mixin import (
    SamplerMixin,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signal import RawSignal
import numpy as np
from sklearn.exceptions import NotFittedError

from copy import deepcopy


class RatspyWrapper(SamplerMixin, RawSignalsAugumenter ):

    def __init__(
        self,
    ) -> None:
        RawSignalsAugumenter.__init__(self)
        SamplerMixin.__init__(self)

    def fit(
        self,
        raw_signals: RawSignals,
    ) -> RawSignalsAugumenter:
        self._is_fitted = True
        return self

    def _check_if_fitted(self):
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise NotFittedError(
                "You must fit the augumenter before calling transform. Call fit() or fit_transform() first."
            )

    def transform(self, raw_signals: RawSignals) -> RawSignals:
        self._check_if_fitted()

        new_signals = raw_signals.initialize_empty()
        
        return new_signals

    def fit_transform(self, raw_signals: RawSignals) -> RawSignals:
        return self.fit(raw_signals).transform(raw_signals)
