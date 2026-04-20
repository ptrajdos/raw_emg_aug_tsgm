from typing import Optional
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
import keras
import tsgm
from tsgm.models.architectures.zoo import Architecture, cGAN_Conv4Architecture
from raw_emg_aug_tsgm.data_augmentation.model_factories.model_factory import (
    ModelFactory,
)


class CGANFactory(ModelFactory):

    def __init__(
        self,
        cgan_architecture_cls: Optional[Architecture] = None,
        cgan_architecture_options: Optional[dict] = None,
        model_construction_options: Optional[dict] = None,
        model_compile_options: Optional[dict] = None,
        latent_dim: int = 32,
    ) -> None:
        super().__init__()
        self.cgan_architecture_cls = cgan_architecture_cls
        self.cgan_architecture_options = cgan_architecture_options
        self.model_construction_params = model_construction_options
        self.model_compile_options = model_compile_options
        self.latent_dim = latent_dim

    def _get_default_architecture_cls(self) -> Architecture:
        return cGAN_Conv4Architecture

    def _get_efective_architecture_cls(self) -> Architecture:
        return (
            self._get_default_architecture_cls()
            if self.cgan_architecture_cls is None
            else self.cgan_architecture_cls
        )  # type: ignore

    def _get_effective_architecture_options(self):
        return (
            {}
            if self.cgan_architecture_options is None
            else self.cgan_architecture_options
        )

    def _get_effective_model_construction_options(self):
        return (
            {}
            if self.model_construction_params is None
            else self.model_construction_params
        )

    def _get_default_model_compile_options(self) -> dict:
        return {
            "d_optimizer": keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            "g_optimizer": keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            "loss_fn": keras.losses.BinaryCrossentropy(from_logits=True),
        }

    def _get_effective_model_compile_options(self):
        return (
            self.model_compile_options
            if self.model_compile_options is not None
            else self._get_default_model_compile_options()
        )

    def get_latent_dim(self) -> int:
        return self.latent_dim

    def _get_architecture_obj(self, raw_signals: RawSignals) -> Architecture:
        dim_params = self._get_dims(raw_signals)
        architecture_construction_params = self._get_effective_architecture_options()
        architecture_construction_params.update(dim_params)

        architecture_obj = self._get_efective_architecture_cls()(
            **architecture_construction_params
        )  # type: ignore
        return architecture_obj
    
    def _get_model_creation_cls(self):
        return tsgm.models.cgan.ConditionalGAN

    
    def _get_model(self, architecture_obj):
        eff_model_constr_options = self._get_effective_model_construction_options()
        eff_model_constr_options.update(architecture_obj.get())
        eff_model_constr_options.update({"latent_dim": self.get_latent_dim()})

        cgan_model: keras.Model = self._get_model_creation_cls()(
            **eff_model_constr_options
        )
        return cgan_model

    def get_compiled_model(self, raw_signals: RawSignals) -> keras.Model:

        architecture_obj = self._get_architecture_obj(raw_signals=raw_signals)

        cgan_model = self._get_model(architecture_obj)

        eff_model_compile_options = self._get_effective_model_compile_options()
        cgan_model.compile(**eff_model_compile_options)

        # TODO update?
        return cgan_model
