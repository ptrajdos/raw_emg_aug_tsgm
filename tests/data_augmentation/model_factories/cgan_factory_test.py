from raw_emg_aug_tsgm.data_augmentation.model_factories.cgan_factory import CGANFactory
from tests.data_augmentation.model_factories.model_factory_test import ModelFactoryTest


class CGANFactoryTest(ModelFactoryTest):

    __test__ = True
    
    def get_factories(self) -> dict:
        return {
            "Base": CGANFactory(),
        }
    
    def _check_compiled(self, model):
        field_names = ["d_optimizer", "g_optimizer", "loss_fn"]
        for field_name in field_names:
            optim = hasattr(model, field_name)
            self.assertTrue(optim, f"Model has no {field_name} field")