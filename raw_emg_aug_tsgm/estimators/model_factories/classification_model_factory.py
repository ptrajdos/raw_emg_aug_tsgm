from typing import Optional
import abc

import keras
import numpy as np
from tsgm.models.architectures.zoo import Architecture
from raw_emg_aug_tsgm.estimators.model_factories.model_factory import (
    ModelFactory,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from tsgm.models.architectures.zoo import ConvnArchitecture

class ClassificationModelFactory(ModelFactory):

    def __init__(
        self,
        architecture_cls: Optional[Architecture] = None,
        architecture_options: Optional[dict] = None,
        model_compile_options: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.architecture_cls = architecture_cls
        self.architecture_options = architecture_options
        self.model_compile_options = model_compile_options


    def _get_dims(self, raw_signals: RawSignals) -> dict:
        """
        Get dimensions
        """
        seq_len, feat_dim = raw_signals[0].to_numpy().shape
        u_labels = np.unique( raw_signals.get_labels())
        output_dim = len(u_labels)

        di = {
            "seq_len": seq_len,
            "feat_dim": feat_dim,
            "output_dim": output_dim,
        }
        return di

    def _get_default_architecture_cls(self):
        return ConvnArchitecture

    def _get_efective_architecture_cls(self) -> Architecture:
        return (
            self._get_default_architecture_cls()
            if self.architecture_cls is None
            else self.architecture_cls
        )  # type: ignore

    def _get_effective_architecture_options(self):
        return {"n_conv_blocks": 1} if self.architecture_options is None else self.architecture_options
    

    def _get_default_model_compile_options(self) -> dict:
        return {
            "loss": keras.losses.CategoricalCrossentropy(),
            "optimizer": keras.optimizers.legacy.Adam(),
            "metrics": ["accuracy"],
        }

    def _get_effective_model_compile_options(self):
        return (
            self.model_compile_options
            if self.model_compile_options is not None
            else self._get_default_model_compile_options()
        )

    def _get_architecture_obj(self, raw_signals: RawSignals) -> Architecture:
        dim_params = self._get_dims(raw_signals)
        architecture_construction_params = self._get_effective_architecture_options()
        architecture_construction_params.update(dim_params)

        architecture_obj = self._get_efective_architecture_cls()(
            **architecture_construction_params
        )  # type: ignore
        return architecture_obj


    def _get_model(self, architecture_obj):
    
        keras_model: keras.Model = architecture_obj.model
        return keras_model

    def get_compiled_model(self, raw_signals: RawSignals) -> keras.Model:

        architecture_obj = self._get_architecture_obj(raw_signals=raw_signals)

        keras_model = self._get_model(architecture_obj)

        eff_model_compile_options = self._get_effective_model_compile_options()
        keras_model.compile(**eff_model_compile_options)

        return keras_model
