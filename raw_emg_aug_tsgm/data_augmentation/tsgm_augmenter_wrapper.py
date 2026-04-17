from copy import deepcopy
import inspect
import numpy as np
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter import (
    RawSignalsAugumenter,
)
from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.sampler_mixin import (
    SamplerMixin,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from tsgm.models.augmentations import WindowWarping


class TSGMAugmenterWrapper(
    SamplerMixin, BaseEstimator, TransformerMixin, RawSignalsAugumenter
):

    def __init__(
        self,
        augmenter_cls=WindowWarping,
        augmenter_gen_params=None,
        per_feature: bool = True,
    ) -> None:
        RawSignalsAugumenter.__init__(self)
        SamplerMixin.__init__(self)

        self.augmenter_cls = augmenter_cls
        self.augmenter_gen_params = augmenter_gen_params
        self.per_feature = per_feature

        self._init_augmenter()

    def _init_augmenter(self):
        if self.augmenter_gen_params is not None:
            for k, v in self.augmenter_gen_params.items():
                setattr(self, f"gen_{k}", v)

    def _reconstruct_gen_params(self):
        params = {
            k: getattr(self, k)
            for k in self.get_params()
            if k not in ("augmenter_cls", "augmenter_gen_params", "per_feature", "n_transform_samples")
        }
        return params
    
    def _has_per_feature(self)->bool:
        signature  = inspect.signature(self.augmenter_cls.__init__)
        return 'per_feature' in signature.parameters

    def fit(
        self,
        raw_signals: RawSignals,
    ) -> RawSignalsAugumenter:
        self._is_fitted = True

        if self._has_per_feature():
            self.augmenter_ = self.augmenter_cls(per_feature = self.per_feature)
        else:
            self.augmenter_ = self.augmenter_cls()

        return self

    def _check_if_fitted(self):
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise NotFittedError(
                "You must fit the augumenter before calling transform. Call fit() or fit_transform() first."
            )

    def transform(self, raw_signals: RawSignals) -> RawSignals:
        self._check_if_fitted()

        rec_params = self._reconstruct_gen_params()
        
        new_signals = raw_signals.initialize_empty()
        for rs in raw_signals:
            new_raw = deepcopy(rs)
            new_raw_np = new_raw.to_numpy()
            new_dtype = new_raw_np.dtype

            reformatted_np = new_raw_np[np.newaxis, :, :]
            aug_raw = self.augmenter_.generate(X = reformatted_np, n_samples=1,**rec_params)
            new_raw.signal = aug_raw[0].astype(new_dtype)
            new_signals.append(new_raw)
        
        return new_signals

    def fit_transform(self, raw_signals: RawSignals) -> RawSignals:
        return self.fit(raw_signals).transform(raw_signals)
