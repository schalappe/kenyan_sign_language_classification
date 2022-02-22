# -*- coding: utf-8 -*-
"""
Set of class for fine tune
"""
import tensorflow as tf

from .addons import models
from .head_net import NormHeadNetV2


class FineTuneModel:
    @staticmethod
    def build(model_name: str, dims: tuple, num_class: int, hidden_unit):
        # load reference model
        head = models[model_name](
            input_shape=dims,
            include_top=False,
            weights="imagenet",
        )

        # Freeze the pretrained weights
        head.trainable = False

        # Add top to reference model
        outputs = NormHeadNetV2.build(
            base_model=head, len_class=num_class, dense_unit=hidden_unit
        )

        return tf.keras.Model(head.input, outputs)
