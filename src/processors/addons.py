# -*- coding: utf-8 -*-
"""
    All necessary for feature
"""
import tensorflow as tf

# model preprocess
model_preprocess = {
    "DenseNet": tf.keras.applications.densenet.preprocess_input,
    "EfficientNet": tf.keras.applications.efficientnet.preprocess_input,
    "NasNet": tf.keras.applications.nasnet.preprocess_input,
    "ResNet": tf.keras.applications.resnet_v2.preprocess_input,
}
