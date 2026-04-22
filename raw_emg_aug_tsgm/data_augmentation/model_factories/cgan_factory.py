import keras
import tsgm
import tsgm.models
from tsgm.models.architectures.zoo import Architecture, cGAN_Conv4Architecture
import tsgm.models.cgan
from raw_emg_aug_tsgm.data_augmentation.model_factories.conditional_model_factory import (
    ConditionalModelFactory,
)


class CGANFactory(ConditionalModelFactory):

    def _get_default_architecture_cls(self) -> Architecture:
        return cGAN_Conv4Architecture

    def _get_default_model_compile_options(self) -> dict:
        return {
            "d_optimizer": keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            "g_optimizer": keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            "loss_fn": keras.losses.BinaryCrossentropy(),
        }
    def _get_default_model_construction_options(self) -> dict:
        return dict()

    def _get_model_creation_cls(self):
        return tsgm.models.cgan.ConditionalGAN
