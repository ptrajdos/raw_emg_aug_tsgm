from raw_emg_aug_tsgm.data_augmentation.model_factories.cgan_factory import CGANFactory
from tests.data_augmentation.model_factories.model_factory_test import ModelFactoryTest
from tsgm.models.architectures import (
    cGAN_LSTMConv3Architecture,
    cGAN_LSTMnArchitecture,
    tcGAN_Conv4Architecture,
)


class CGANFactoryTest(ModelFactoryTest):

    __test__ = True

    def get_factories(self) -> dict:
        return {
            "Base": CGANFactory(),
            "LSTM-CONV": CGANFactory(architecture_cls=cGAN_LSTMConv3Architecture),
            "LSTMn": CGANFactory(architecture_cls=cGAN_LSTMnArchitecture),
            # "tcGAN": CGANFactory(
            #     architecture_cls=tcGAN_Conv4Architecture,                
            #     model_construction_options={"temporal": True},
            #     latent_dim=1
            # ),  # TODO incompatible 
        }

    def _check_compiled(self, model):
        field_names = ["d_optimizer", "g_optimizer", "loss_fn"]
        for field_name in field_names:
            optim = hasattr(model, field_name)
            self.assertTrue(optim, f"Model has no {field_name} field")
