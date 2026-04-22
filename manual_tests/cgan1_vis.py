from dexterous_bioprosthesis_2021_raw_datasets.raw_signals_creators.raw_signals_creator_sines import (
    RawSignalsCreatorSines,
)
import keras
import matplotlib.pyplot as plt
from dexterous_bioprosthesis_2021_raw_datasets.raw_signals.raw_signals import RawSignals
import numpy as np

from raw_emg_aug_tsgm.data_augmentation.model_factories.cgan_factory import CGANFactory
from raw_emg_aug_tsgm.data_augmentation.tsgm_nn_augmenter_wrapper import (
    TSGMANNAugmenterWrapper,
)
from tsgm.metrics.metrics import MMDMetric

from tsgm.models.architectures import (
    cGAN_LSTMConv3Architecture,
    cGAN_LSTMnArchitecture,
    tcGAN_Conv4Architecture,
)
from tensorflow.keras import mixed_precision
import tensorflow as tf
from matplotlib.backends.backend_pdf import PdfPages
# tf.keras.backend.set_floatx('float16')

# mixed_precision.set_global_policy('mixed_float16')

N, R, C = 500, 50, 1

set_creat = RawSignalsCreatorSines(
    set_size=N, column_number=C, samples_number=R, dtype=np.float16
)
raw_data: RawSignals = set_creat.get_set()
rep_file_path = "vis.md"

with PdfPages('cgan1_vis.pdf') as pdf, open(rep_file_path, "w") as rep_file:
    rd = raw_data[0].to_numpy()
    plt.plot(rd)
    plt.title("Original signal")
    pdf.savefig()
    plt.close()

    opts = {
        "d_optimizer": keras.optimizers.legacy.Adam(learning_rate=0.0002, beta_1=0.5),
        "g_optimizer": keras.optimizers.legacy.Adam(learning_rate=0.0002, beta_1=0.5),
        "loss_fn": keras.losses.BinaryCrossentropy(),
    }

    gen = TSGMANNAugmenterWrapper(
        model_factory=CGANFactory(model_compile_options=opts, architecture_cls=cGAN_LSTMnArchitecture),
        normalize_channels=True,
        forced_keras_dtype=np.float16,
    )
    gen.fit(raw_data, epochs=10, batch_size=32)

    generted = gen.transform(raw_data)

    r = generted[0].to_numpy()

    mmd_metric = MMDMetric()
    mmd_value = mmd_metric(raw_data.to_numpy(), generted.to_numpy())
    rep_file.write(f"MMD between original and generated data: {mmd_value}\n")



    plt.plot(r)
    plt.title("Generated signal")
    pdf.savefig()
    plt.close()
    

