from raw_emg_aug_tsgm.data_augmentation.model_factories.cgan_factory import CGANFactory
from raw_emg_aug_tsgm.data_augmentation.tsgm_nn_augmenter_wrapper import TSGMANNAugmenterWrapper
from tests.data_augmentation.raw_signals_augumenter_test import RawSignalsAugumenterTest


class RawSignalsAugumenterDummyTest(RawSignalsAugumenterTest):


    __test__ = True

    def get_augumenters(self):
        return {
            "Base": TSGMANNAugmenterWrapper(model_factory=CGANFactory())
        }