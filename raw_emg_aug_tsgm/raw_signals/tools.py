import numpy as np

from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import (
    RawSignals,
)


def convert_to_tensor(raw_signals: RawSignals) -> np.ndarray:
    np_list = [rs.to_numpy() for rs in raw_signals]
    result = np.stack(np_list, axis=0)
    return result
