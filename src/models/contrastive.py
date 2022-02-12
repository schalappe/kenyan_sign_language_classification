# -*- coding: utf-8 -*-
"""
Set of function for supervised contrastive learning
"""
import tensorflow as tf

from src.addons import npairs_multilabel_loss


class SupervisedContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)

        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return npairs_multilabel_loss(tf.squeeze(labels), logits)


def create_encoder(model: tf.keras.Model, input_shape: tuple):
    inputs = tf.keras.Input(shape=input_shape)
    outputs = model(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="encoder")
    return model


def create_classifier(
    encoder: tf.keras.Model,
    input_shape: tuple,
    dropout_rate: float,
    hidden_units: int,
    num_classes: int,
    trainable=True,
):
    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = tf.keras.Input(shape=input_shape)
    features = encoder(inputs)
    features = tf.keras.layers.Dropout(dropout_rate)(features)
    features = tf.keras.layers.Dense(hidden_units, activation="relu")(features)
    features = tf.keras.layers.Dropout(dropout_rate)(features)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(features)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="classifier")
    return model


def add_projection_head(
    encoder: tf.keras.Model, input_shape: tuple, projection_units: int
):
    inputs = tf.keras.Input(shape=input_shape)
    features = encoder(inputs)
    outputs = tf.keras.layers.Dense(projection_units, activation="relu")(features)
    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name="encoder_with_projection-head"
    )
    return model
