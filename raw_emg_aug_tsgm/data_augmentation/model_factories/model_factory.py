import abc
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
import keras


class ModelFactory(abc.ABC):


    @abc.abstractmethod
    def get_compiled_model(self, raw_signals: RawSignals)->keras.Model:
        """
        Get TGAN compiled model.
        """
