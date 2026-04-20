from raw_emg_aug_tsgm.data_augmentation.model_factories.cvae_factory import CVAEFactory
from tests.data_augmentation.model_factories.model_factory_test import ModelFactoryTest


class CVAEFactoryTest(ModelFactoryTest):

    __test__ = True

    def get_factories(self) -> dict:
        return {
            "Base": CVAEFactory(),
        }

    def _check_compiled(self, model):
        pass
