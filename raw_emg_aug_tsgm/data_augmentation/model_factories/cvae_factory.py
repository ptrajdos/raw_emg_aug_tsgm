import keras
import tsgm
import tsgm.models
from tsgm.models.architectures.zoo import Architecture, cVAE_CONV5Architecture
import tsgm.models.cvae
from raw_emg_aug_tsgm.data_augmentation.model_factories.conditional_model_factory import (
    ConditionalModelFactory,
)


class CVAEFactory(ConditionalModelFactory):

    def _get_default_architecture_cls(self) -> Architecture:
        return cVAE_CONV5Architecture

    def _get_default_model_compile_options(self) -> dict:
        return {
            "optimizer": keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        }

    def _get_default_model_construction_options(self) -> dict:
        return {"temporal":False}
    
    def _get_model_creation_cls(self):
        return tsgm.models.cvae.cBetaVAE
