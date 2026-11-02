"""Downstream intrusion-detection classifier (same model for baseline & augmented runs)."""
import tensorflow as tf
from tensorflow.keras import Sequential, layers


def build_classifier(n_features, n_classes):
    model = Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
