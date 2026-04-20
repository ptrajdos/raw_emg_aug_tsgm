import unittest

import keras

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_creators.raw_signals_creator_sines import (
    RawSignalsCreatorSines,
)

from raw_emg_aug_tsgm.raw_signals.tools import convert_to_tensor


class ModelFactoryTest(unittest.TestCase):

    __test__ = False

    @classmethod
    def setUpClass(cls):
        if not cls.__test__:
            raise unittest.SkipTest("Skipping")

    def get_factories(self) -> dict:
        raise unittest.SkipTest("Skipping")
    
    def gen_data(self, N=10, T=100, C=3):
        generator = RawSignalsCreatorSines(set_size=N, samples_number=T, column_number=C)
        return generator.get_set()


    def _check_compiled(self, model):
        pass

    def test_types(self):
    
        raw_data = self.gen_data()
        for factory_name, factory in self.get_factories().items():

            with self.subTest(factory_name=factory_name):
                model = factory.get_compiled_model(raw_data)

                self.assertIsNotNone(model, "Model is None")
                self.assertIsInstance(model, keras.Model, "Wrong model type")
                self._check_compiled(model)

    def test_fitable(self):
    
        raw_data = self.gen_data()
        for factory_name, factory in self.get_factories().items():

            with self.subTest(factory_name=factory_name):
                model = factory.get_compiled_model(raw_data)
                np_data = convert_to_tensor(raw_data)
                y = raw_data.get_labels().reshape(-1,1)
                model.fit(np_data, y)

