import abc
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals


class ModelFactory(abc.ABC):

    def get_output_dim(self) -> int:
        return 1

    def get_latent_dim(self) -> int:
        return 32

    def _get_dims(self, raw_signals: RawSignals) -> dict:
        """
        Get dimensions
        """
        seq_len, feat_dim = raw_signals[0].to_numpy().shape
        latent_dim = self.get_latent_dim()
        output_dim = self.get_output_dim()

        di = {
            "seq_len": seq_len,
            "feat_dim": feat_dim,
            "latent_dim": latent_dim,
            "output_dim": output_dim,
        }
        return di

    @abc.abstractmethod
    def get_compiled_model(self, raw_signals: RawSignals):
        """
        Get TGAN compiled model.
        """
