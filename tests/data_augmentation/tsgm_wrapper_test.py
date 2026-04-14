
from raw_emg_aug_tsgm.data_augmentation.tsgm_wrapper import RatspyWrapper
from tests.data_augmentation.raw_signals_augumenter_test import RawSignalsAugumenterTest


class RawSignalsAugumenterDummyTest(RawSignalsAugumenterTest):


    __test__ = True

    def get_augumenter(self):
        return RatspyWrapper()