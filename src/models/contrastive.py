# -*- coding: utf-8 -*-
"""
Set of function for supervised contrastive learning
"""
import tensorflow as tf

from src.addons import npairs_multilabel_loss

from .head_net import NormHeadNetV2


class SupervisedContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=1, name=None) -> None:
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None) -> tf.Tensor:
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


def create_encoder(model: tf.keras.Model, input_shape: tuple, last_layers: float) -> tf.keras.Model:
    """
    Create an encoder
    Parameters
    ----------
    model: tf.keras.Model
        Head of encoder

    input_shape: tuple
        Dimension of input images

    last_layers: float
        Percentage of layers to train

    Returns
    -------

    """
    # Create encoder
    inputs = tf.keras.Input(shape=input_shape)
    outputs = model(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="encoder")

    # unfreeze layers
    last_layers = int(last_layers * len(model.layers))
    for layer in model.layers[:last_layers]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    return model


def create_classifier(
    encoder: tf.keras.Model,
    input_shape: tuple,
    dropout_rate: float,
    hidden_units: int,
    num_classes: int,
    trainable=True,
) -> tf.keras.Model:
    """
    Create a classifier
    Parameters
    ----------
    encoder: tf.keras.Model
        Header of model

    input_shape: tuple
        Dimension of input image

    dropout_rate: float
        Rate of dropout

    hidden_units: int
        Number of hidden neurons

    num_classes: int
        Number of class

    trainable: bool
        Train layers or not

    Returns
    -------

    """
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


def create_classifier_v2(
    encoder: tf.keras.Model,
    input_shape: tuple,
    hidden_units: int,
    num_classes: int,
    trainable=True,
) -> tf.keras.Model:
    """
    Create a classifier
    Parameters
    ----------
    encoder: tf.keras.Model
        Header of model

    input_shape: tuple
        Dimension of input image

    dropout_rate: float
        Rate of dropout

    hidden_units: int
        Number of hidden neurons

    num_classes: int
        Number of class

    trainable: bool
        Train layers or not

    Returns
    -------

    """
    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = tf.keras.Input(shape=input_shape)
    features = encoder(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(features)
    x = tf.keras.layers.Dropout(0.1, name="top_dropout_1")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(hidden_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5, name="top_dropout_2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="classifier")
    return model


def add_projection_head(
    encoder: tf.keras.Model, input_shape: tuple, projection_units: int
) -> tf.keras.Model:
    """
    Add projection to encoder

    Parameters
    ----------
    encoder: tf.keras.Model

    input_shape: tuple
        Dimension of input image

    projection_units: int
        length of encode image

    Returns
    -------

    """
    inputs = tf.keras.Input(shape=input_shape)
    features = encoder(inputs)
    outputs = tf.keras.layers.Dense(projection_units, activation="relu")(features)
    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name="encoder_with_projection_head"
    )
    return model
