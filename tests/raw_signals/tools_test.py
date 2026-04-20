import unittest

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_creators.raw_signals_creator_sines import (
    RawSignalsCreatorSines,
)
import numpy as np
from raw_emg_aug_tsgm.raw_signals.tools import convert_to_tensor


class ToolsTest(unittest.TestCase):

    def gen_data(self, N=10, T=100, C=3):
        generator = RawSignalsCreatorSines(
            set_size=N, samples_number=T, column_number=C
        )
        return generator.get_set()

    def test_to_tensor(self):
        raw_set = self.gen_data()
        rs_dtype = raw_set[0].to_numpy().dtype
        R, C = raw_set[0].to_numpy().shape

        tensor = convert_to_tensor(raw_signals=raw_set)
        self.assertIsNotNone(tensor, "Tensor is None")
        self.assertIsInstance(tensor, np.ndarray, "Wrong tensor type")
        self.assertTrue(
            tensor.dtype == rs_dtype,
            f"Wrong signal dtype expect {rs_dtype} got {tensor.dtype}",
        )

        self.assertTrue(len(tensor.shape) == 3, "Wrong number of axes in tensor")
        self.assertTrue(len(raw_set) == tensor.shape[0], "Wrong number of instances")
        self.assertTrue(R == tensor.shape[1], "Wrong number of samples")
        self.assertTrue(C == tensor.shape[2], "Wrong number of columns")
