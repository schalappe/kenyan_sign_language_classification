# -*- coding: utf-8 -*-
"""
    All necessary for training
"""
import tensorflow as tf

models = {
    "DenseNet121": tf.keras.applications.DenseNet121,
    "DenseNet169": tf.keras.applications.DenseNet169,
    "DenseNet201": tf.keras.applications.DenseNet201,
    "EfficientNetB7": tf.keras.applications.EfficientNetB7,
    "NasNetMobile": tf.keras.applications.NASNetMobile,
    "ResNet50V2": tf.keras.applications.ResNet50V2,
    "ResNet101V2": tf.keras.applications.ResNet101V2,
    "ResNet152V2": tf.keras.applications.ResNet152V2,
}
