import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Reshape
import matplotlib.pyplot as plt
import util
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
X_train = util.X_train.reshape(util.X_train.shape[0], util.X_train.shape[1], 1)
X_test = util.X_test.reshape(util.X_test.shape[0], util.X_test.shape[1], 1)


# Define the VAE model
class VAE(keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        inputs = keras.Input(shape=(11, 1))  # Ensure the input shape matches your data
        x = layers.Flatten()(inputs)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)

        # Define latent space mean and log variance
        # z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        # z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        flatten = layers.Flatten()(x)
        mean = layers.Dense(2, name='z_mean')(flatten)
        log_var = layers.Dense(2, name='z_log_var')(flatten)
        model = tf.keras.Model(inputs, (mean, log_var), name="Encoder")
        return model

    def reparameterize(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], self.latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def build_decoder(self):
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(32, activation='relu')(latent_inputs)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        decoder_outputs = layers.Dense(11)(x)

        return keras.Model(latent_inputs, decoder_outputs)

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decoder(z)
        return reconstructed


# Define the loss function for VAE
def vae_loss(inputs, reconstructed):
    z_mean, z_log_var = vae.encoder(inputs)  # Get z_mean and z_log_var from the model
    reconstruction_loss = keras.losses.mean_squared_error(inputs, reconstructed)
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    return reconstruction_loss + kl_loss
"""
def vae_loss(inputs, reconstructed):
    z_mean, z_log_var = vae.encoder(inputs)  # Get z_mean and z_log_var from the model

    # Reshape inputs to match the shape of the reconstructed data
    inputs_reshaped = tf.reshape(inputs, (-1, 11))

    reconstruction_loss = keras.losses.mean_squared_error(inputs_reshaped, reconstructed)
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    return reconstruction_loss + kl_loss
"""
# Ensure you pass the correct input data to the loss function

# Create and compile the VAE model
# Create and compile the VAE model
latent_dim = 2  # Set your desired latent dimension
vae = VAE(latent_dim=latent_dim)
vae.compile(optimizer='adam', loss=lambda inputs, reconstructed: vae_loss(inputs, reconstructed))

# Generate new time series data using X_test

# Train the VAE model on the training data
history = vae.fit(X_train, X_train, epochs=10, batch_size=32)

# Create an instance of the VAE model for summary
model_summary = VAE(latent_dim=latent_dim)
model_summary.build((None, 11, 1))  # Specify input shape for the summary
model_summary.call(tf.convert_to_tensor(X_train))  # Call with an example input to build the model

# Print the model summary
model_summary.summary()
