"""Auxiliary Classifier GAN (ACGAN) for tabular network-traffic data.

Generator     : (noise, class label) -> synthetic feature vector
Discriminator : feature vector -> (real/fake, traffic class)   <- dual task
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers


def build_generator(latent_dim, n_classes, n_features):
    noise = layers.Input(shape=(latent_dim,))
    label = layers.Input(shape=(1,), dtype="int32")
    emb = layers.Flatten()(layers.Embedding(n_classes, latent_dim)(label))
    x = layers.Multiply()([noise, emb])
    for units in (128, 256, 512):
        x = layers.Dense(units)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
    out = layers.Dense(n_features, activation="sigmoid")(x)
    return Model([noise, label], out, name="generator")


def build_discriminator(n_features, n_classes):
    inp = layers.Input(shape=(n_features,))
    x = inp
    for units in (512, 256, 128):
        x = layers.Dense(units)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
    validity = layers.Dense(1, activation="sigmoid", name="validity")(x)
    cls = layers.Dense(n_classes, activation="softmax", name="class")(x)
    return Model(inp, [validity, cls], name="discriminator")


class ACGAN:
    def __init__(self, n_features, n_classes, latent_dim=100, lr=2e-4):
        self.n_features, self.n_classes, self.latent_dim = n_features, n_classes, latent_dim
        self.generator = build_generator(latent_dim, n_classes, n_features)
        self.discriminator = build_discriminator(n_features, n_classes)
        self.g_opt = tf.keras.optimizers.Adam(lr, beta_1=0.5)
        self.d_opt = tf.keras.optimizers.Adam(lr, beta_1=0.5)
        self.bce = tf.keras.losses.BinaryCrossentropy()
        self.scce = tf.keras.losses.SparseCategoricalCrossentropy()

    @tf.function
    def _train_step(self, real_x, real_y):
        bs = tf.shape(real_x)[0]
        real_y = tf.reshape(real_y, (-1, 1))
        noise = tf.random.normal((bs, self.latent_dim))
        # sample classes uniformly so minority classes get equal attention
        fake_y = tf.random.uniform((bs, 1), 0, self.n_classes, dtype=tf.int32)
        ones, zeros = tf.ones((bs, 1)), tf.zeros((bs, 1))

        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            fake_x = self.generator([noise, fake_y], training=True)
            r_valid, r_cls = self.discriminator(real_x, training=True)
            f_valid, f_cls = self.discriminator(fake_x, training=True)

            d_loss = (self.bce(ones * 0.9, r_valid) + self.scce(real_y, r_cls)
                      + self.bce(zeros, f_valid) + self.scce(fake_y, f_cls))
            g_loss = self.bce(ones, f_valid) + self.scce(fake_y, f_cls)

        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.d_opt.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
        self.g_opt.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        return d_loss, g_loss

    def fit(self, X, y, epochs=100, batch_size=128, log_every=10):
        ds = (tf.data.Dataset.from_tensor_slices((X, y))
              .shuffle(len(X)).batch(batch_size, drop_remainder=True)
              .prefetch(tf.data.AUTOTUNE))
        history = {"d_loss": [], "g_loss": []}
        for epoch in range(1, epochs + 1):
            d_losses, g_losses = [], []
            for bx, by in ds:
                d, g = self._train_step(bx, by)
                d_losses.append(float(d))
                g_losses.append(float(g))
            history["d_loss"].append(np.mean(d_losses))
            history["g_loss"].append(np.mean(g_losses))
            if epoch % log_every == 0 or epoch == 1:
                print(f"[ACGAN] epoch {epoch:>3}/{epochs}  "
                      f"D_loss={history['d_loss'][-1]:.4f}  G_loss={history['g_loss'][-1]:.4f}")
        return history

    def generate(self, class_id, n):
        noise = np.random.normal(size=(n, self.latent_dim)).astype("float32")
        labels = np.full((n, 1), class_id, dtype="int32")
        return self.generator.predict([noise, labels], verbose=0)
