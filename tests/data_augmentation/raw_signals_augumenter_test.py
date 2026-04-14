import unittest

import numpy as np
from sklearn.exceptions import NotFittedError

from dexterous_bioprosthesis_2021_raw_datasets.data_augumentation.raw_signals_augumenter import (
    RawSignalsAugumenter,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_creators.raw_signals_creator_sines import (
    RawSignalsCreatorSines,
)
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_creators.raw_signals_creator_zeros import (
    RawSignalsCreatorZeros,
)


class RawSignalsAugumenterTest(unittest.TestCase):

    __test__ = False

    @classmethod
    def setUpClass(cls):
        if not cls.__test__:
            raise unittest.SkipTest("Skipping")

    def get_augumenter(self) -> RawSignalsAugumenter:
        raise unittest.SkipTest("Skipping")

    def _check_aug_signals(self, raw_signals: RawSignals, aug_signals: RawSignals):
        self.assertIsNotNone(aug_signals, "Augumented signals none.")
        self.assertTrue(len(aug_signals) >= 1, "Number of augumented points.")
        self.assertIsInstance(
            aug_signals, RawSignals, "Wrong type of the returned object"
        )
        self.assertTrue(
            aug_signals.get_sample_rate() == raw_signals.get_sample_rate(),
            "Wrong sample rate",
        )
        self.assertTrue(
            aug_signals.signal_n_cols == raw_signals.signal_n_cols,
            "Wrong number of columns",
        )

        for a_sig in aug_signals:
            self.assertFalse(
                np.isnan(a_sig.to_numpy()).any(), "Nans in transformed signals"
            )
            self.assertFalse(
                np.isinf(a_sig.to_numpy()).any(), "Infs in transformed signals"
            )

    def test_fit_then_transform(self):
        signal_creator = RawSignalsCreatorSines(samples_number=1000)
        raw_signals: RawSignals = signal_creator.get_set()

        aug = self.get_augumenter()

        obj = aug.fit(raw_signals)
        self.assertIsNotNone(obj, "fit should return something")
        self.assertTrue(type(obj) == type(aug), "Fit should return self")
        aug_signals: RawSignals = aug.transform(raw_signals)

        self._check_aug_signals(raw_signals=raw_signals, aug_signals=aug_signals)

    def test_fit_then_transform_zeros(self):
        signal_creator = RawSignalsCreatorZeros(samples_number=1000)
        raw_signals: RawSignals = signal_creator.get_set()

        aug = self.get_augumenter()

        obj = aug.fit(raw_signals)
        self.assertIsNotNone(obj, "fit should return something")
        self.assertTrue(type(obj) == type(aug), "Fit should return self")
        aug_signals: RawSignals = aug.transform(raw_signals)

        self._check_aug_signals(raw_signals=raw_signals, aug_signals=aug_signals)

    def test_fit_transform(self):
        signal_creator = RawSignalsCreatorSines()
        raw_signals = signal_creator.get_set()

        aug = self.get_augumenter()

        aug_signals = aug.fit_transform(raw_signals)

        self._check_aug_signals(raw_signals, aug_signals)

    def test_not_fitted(self):
        signal_creator = RawSignalsCreatorSines()
        raw_signals = signal_creator.get_set()

        aug = self.get_augumenter()

        with self.assertRaises(NotFittedError):
            aug.transform(raw_signals)

    def test_not_fitted_sample(self):
        signal_creator = RawSignalsCreatorSines()
        raw_signals = signal_creator.get_set()

        aug = self.get_augumenter()

        with self.assertRaises(NotFittedError):
            aug.sample(raw_signals)

    def test_sample(self):
        signal_creator = RawSignalsCreatorSines()
        raw_signals = signal_creator.get_set()
        n_raw_signals = len(raw_signals)

        aug = self.get_augumenter()
        aug.fit(raw_signals)

        samples_to_select = (1, 2, 3, 10, n_raw_signals, n_raw_signals + 10)

        for n_samples in samples_to_select:
            with self.subTest(n_samples=n_samples):
                sampled_signals = aug.sample(raw_signals, n_samples=n_samples)

                self._check_aug_signals(raw_signals, sampled_signals)
                self.assertTrue(
                    len(sampled_signals) == n_samples,
                    f"Wrong number of sampled points. Expected {n_samples}, got {len(sampled_signals)}",
                )

    def test_dtype(self):
        dtypes = [np.float32, np.float64, np.single, np.double]
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                signal_creator = RawSignalsCreatorSines(dtype=dtype)
                raw_signals = signal_creator.get_set()

                aug = self.get_augumenter()
                aug_signals = aug.fit_transform(raw_signals)

                self._check_aug_signals(raw_signals, aug_signals)

                for sig in aug_signals:
                    self.assertTrue(
                        np.issubdtype(sig.to_numpy().dtype, dtype),
                        f"Wrong dtype of the signal. Expected {dtype}, got {sig.to_numpy().dtype}   ",
                    )

    def test_sample_dtype(self):
        dtypes = [np.float32, np.float64, np.single, np.double]
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                signal_creator = RawSignalsCreatorSines(dtype=dtype)
                raw_signals = signal_creator.get_set()

                aug = self.get_augumenter()
                aug.fit(raw_signals)

                sampled_signals = aug.sample(raw_signals, n_samples=10)

                self._check_aug_signals(raw_signals, sampled_signals)

                for sig in sampled_signals:
                    self.assertTrue(
                        np.issubdtype(sig.to_numpy().dtype, dtype),
                        f"Wrong dtype of the signal. Expected {dtype}, got {sig.to_numpy().dtype}   ",
                    )

if __name__ == "__main__":
    unittest.main()
