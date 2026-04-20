from typing import Optional
import abc

import keras
from tsgm.models.architectures.zoo import Architecture
from raw_emg_aug_tsgm.data_augmentation.model_factories.model_factory import (
    ModelFactory,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals


class ConditionalModelFactory(ModelFactory):

    def __init__(
        self,
        architecture_cls: Optional[Architecture] = None,
        architecture_options: Optional[dict] = None,
        model_construction_options: Optional[dict] = None,
        model_compile_options: Optional[dict] = None,
        latent_dim: int = 32,
    ) -> None:
        super().__init__()
        self.architecture_cls = architecture_cls
        self.architecture_options = architecture_options
        self.model_construction_params = model_construction_options
        self.model_compile_options = model_compile_options
        self.latent_dim = latent_dim

    def get_output_dim(self) -> int:
        return 1

    def get_latent_dim(self) -> int:
        return self.latent_dim

    def _get_dims(self, raw_signals: RawSignals) -> dict:
        """
        Get dimensions
        """
        seq_len, feat_dim = raw_signals[0].to_numpy().shape
        latent_dim = self.get_latent_dim()
        output_dim = self.get_output_dim()

        di = {
            "seq_len": seq_len,
            "feat_dim": feat_dim,
            "latent_dim": latent_dim,
            "output_dim": output_dim,
        }
        return di

    @abc.abstractmethod
    def _get_default_architecture_cls(self):
        pass

    def _get_efective_architecture_cls(self) -> Architecture:
        return (
            self._get_default_architecture_cls()
            if self.architecture_cls is None
            else self.architecture_cls
        )  # type: ignore

    def _get_effective_architecture_options(self):
        return {} if self.architecture_options is None else self.architecture_options
    
    @abc.abstractmethod
    def _get_default_model_construction_options(self)->dict:
        pass

    def _get_effective_model_construction_options(self):
        return (
            self._get_default_model_construction_options()
            if self.model_construction_params is None
            else self.model_construction_params
        )

    @abc.abstractmethod
    def _get_default_model_compile_options(self) -> dict:
        pass

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

    @abc.abstractmethod
    def _get_model_creation_cls(self):
        pass

    def _get_model(self, architecture_obj):
        eff_model_constr_options = self._get_effective_model_construction_options()
        eff_model_constr_options.update(architecture_obj.get())
        eff_model_constr_options.update({"latent_dim": self.get_latent_dim()})

        cgan_model: keras.Model = self._get_model_creation_cls()(
            **eff_model_constr_options
        )  # type: ignore
        return cgan_model

    def get_compiled_model(self, raw_signals: RawSignals) -> keras.Model:

        architecture_obj = self._get_architecture_obj(raw_signals=raw_signals)

        cgan_model = self._get_model(architecture_obj)

        eff_model_compile_options = self._get_effective_model_compile_options()
        cgan_model.compile(**eff_model_compile_options)

        return cgan_model
