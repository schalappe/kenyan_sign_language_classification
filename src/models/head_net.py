# -*- coding: utf-8 -*-
"""
Set of head net for transfer learning
"""
from typing import Any

import tensorflow as tf


class FCHeadNet:
    @staticmethod
    def build(base_model: Any, len_class: int, dense_unit: int) -> Any:
        """
        Fully connected header
        Parameters
        ----------
        base_model: Any
            Base model

        len_class: int
            length of class

        dense_unit: int
            Number of unit in fully connected layer

        Returns
        -------
        Any:
            Head of network
        """
        # initialize the head model that will be placed on top of
        # the base, then add a FC layer
        x = tf.keras.layers.Flatten(name="flatten")(base_model.output)
        x = tf.keras.layers.Dense(dense_unit, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        # add a softmax layer
        outputs = tf.keras.layers.Dense(len_class, activation="softmax")(x)

        return outputs


class NormHeadNet:
    @staticmethod
    def build(base_model: Any, len_class: int) -> Any:
        """
        Batch Normalization header
        Parameters
        ----------
        base_model: Any
            Base model

        len_class: int
            length of clas

        Returns
        -------
        Any:
            Head of network
        """
        # initialize the head model that will be placed on top of
        # the base, then add a Normalization layer
        x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2, name="top_dropout")(x)

        # add a softmax layer
        outputs = tf.keras.layers.Dense(len_class, activation="softmax")(x)

        return outputs
