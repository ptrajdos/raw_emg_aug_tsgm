import os
import numpy as np

# 1. Backend Configuration
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import tsgm

# 2. Parameters & Dummy Data
n_samples, seq_len, feat_dim = 200, 64, 8
n_classes = 3
latent_dim = 32

# Create synthetic data
X = np.random.standard_normal((n_samples, seq_len, feat_dim)).astype(np.float32)
# FIX: Ensure labels are 2D (n_samples, 1) to avoid IndexError
y = np.random.randint(0, n_classes, size=(n_samples, 1)).astype(np.float32)

# 3. Preprocessing (Crucial for GAN stability)
# Scale data to [0, 1] range
scaler = tsgm.utils.TSFeatureWiseScaler((-1, 1))
X_scaled = scaler.fit_transform(X)

# 4. Model Architecture from Zoo
# 'cgan_base_c4_l1' expects (seq_len, feat_dim, latent_dim, output_dim)
architecture = tsgm.models.architectures.zoo["cgan_base_c4_l1"](
    seq_len=seq_len, 
    feat_dim=feat_dim, 
    latent_dim=latent_dim, 
    output_dim=1  # We are passing a single label column
)

# 5. Initialize Conditional GAN
cgan = tsgm.models.cgan.ConditionalGAN(
    discriminator=architecture.discriminator,
    generator=architecture.generator,
    latent_dim=latent_dim
)

cgan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

# 6. Training
print("Starting training...")
cgan.fit(X_scaled, y, epochs=20, batch_size=32)

# 7. Targeted Generation
# Generate 5 samples specifically for Class 2
target_labels = np.array([[2]] * 5).astype(np.float32)
generated_samples_scaled = cgan.generate(target_labels)

# Inverse transform to get back to original EMG/sensor scale
generated_samples = scaler.inverse_transform(generated_samples_scaled)

print(f"Generated samples shape: {generated_samples.shape}")
print(f"Sample data range: {np.min(generated_samples):.2f} to {np.max(generated_samples):.2f}")