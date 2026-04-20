from raw_emg_aug_tsgm.data_augmentation.tsgm_augmenter_wrapper import TSGMAugmenterWrapper
from tests.data_augmentation.raw_signals_augumenter_test import RawSignalsAugumenterTest
from tsgm.models.augmentations import GaussianNoise, MagnitudeWarping, WindowWarping

class TSGMAugmenterWrapperTest(RawSignalsAugumenterTest):


    __test__ = True

    def get_augumenters(self):
        return {
            "Base":TSGMAugmenterWrapper(),
            "Window-args":TSGMAugmenterWrapper(augmenter_cls=WindowWarping, augmenter_gen_params={ 'window_ratio':0.2}),
            "GN": TSGMAugmenterWrapper(augmenter_cls=GaussianNoise),
            "MW": TSGMAugmenterWrapper(augmenter_cls=MagnitudeWarping),
        } 
    