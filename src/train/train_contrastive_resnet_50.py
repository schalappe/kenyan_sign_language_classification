# -*- coding: utf-8 -*-
"""
    Script used to train constrastive model: ResNet50
"""
from os.path import join

import tensorflow as tf

from src.config import CLASS_NAMES, DIMS_MODEL, FEATURES_PATH, MODEL_PATH
from src.data import prepare_from_tfrecord
from src.models import (
    SupervisedContrastiveLoss,
    add_projection_head,
    create_classifier,
    create_encoder,
)

# initialize the number of epochs to train for and batch size
temperature = 0.05
LR_init = 1e-3
BS = 128

# load dataset
print("\n[INFO]: Load datase")
train_set = prepare_from_tfrecord(
    tfrecord=join(FEATURES_PATH, "Train_ResNet.tfrecords"), batch=BS, train=True
)

test_set = prepare_from_tfrecord(
    tfrecord=join(FEATURES_PATH, "Test_ResNet.tfrecords"), batch=BS, train=False
)

# construct our encoder
print("\n[INFO]: Create encoder")
resnet = tf.keras.applications.ResNet50V2(
    include_top=False,
    weights=None,
    input_shape=DIMS_MODEL,
    pooling="avg",
)

encoder = create_encoder(model=resnet, input_shape=DIMS_MODEL)

# add projection header to model
print("\n[INFO]: Add projection header")
encoder_with_projection_head = add_projection_head(
    encoder=encoder,
    input_shape=DIMS_MODEL,
    projection_units=128,
)
encoder_with_projection_head.compile(
    optimizer=tf.keras.optimizers.Adam(LR_init),
    loss=SupervisedContrastiveLoss(temperature),
)

# train encoder
print("\n[INFO]: Train encoder")
H = encoder_with_projection_head.fit(
    train_set,
    validation_data=test_set,
    epochs=50,
    steps_per_epoch=39,
    validation_steps=9,
)


# create and train classifier
print("\n[INFO]: Create and Train classifer")
classifier = create_classifier(
    encoder,
    input_shape=DIMS_MODEL,
    dropout_rate=0.5,
    hidden_units=512,
    num_classes=len(CLASS_NAMES),
    trainable=False,
)

classifier.compile(
    optimizer=tf.keras.optimizers.Adam(LR_init),
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"],
)

# callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        join(MODEL_PATH, "best_resnet_contrastive.h5"),
        monitor="val_loss",
        mode="min",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
    ),
]

H = classifier.fit(
    train_set,
    validation_data=test_set,
    epochs=50,
    callbacks=callbacks,
    steps_per_epoch=39,
    validation_steps=9,
)

# show the accuracy on the testing set
(loss, accuracy) = classifier.evaluate(test_set, steps=9)
print(f"[INFO] accuracy after warn up: {accuracy * 100}%")
print(f"[INFO] loss after warn up: {loss}%")
