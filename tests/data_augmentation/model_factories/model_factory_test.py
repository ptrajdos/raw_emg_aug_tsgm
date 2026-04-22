import unittest

import keras

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_creators.raw_signals_creator_sines import (
    RawSignalsCreatorSines,
)
import numpy as np



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
        x_types = [np.float32, np.float64]
        y_types = [np.float32, np.int32] #By default accepts only numeric types
        for factory_name, factory in self.get_factories().items():
            for xtype in x_types:
                for ytype in y_types:

                    with self.subTest(factory_name=factory_name, xtype=xtype, ytype=ytype):
                        model = factory.get_compiled_model(raw_data)
                        np_data = raw_data.to_numpy().astype(xtype)
                        y = raw_data.get_labels().reshape(-1,1).astype(ytype)
                        y = keras.utils.to_categorical(y,3)
                        model.fit(np_data, y)

