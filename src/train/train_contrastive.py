# -*- coding: utf-8 -*-
"""
    Script used to train constrastive model: ResNet50
"""
import argparse
from os.path import join

import tensorflow as tf

from addons import models
from src.config import CLASS_NAMES, DIMS_MODEL, FEATURES_PATH, MODEL_PATH
from src.data import prepare_from_tfrecord, prepare_from_tfrecord_v2
from src.models import (
    SupervisedContrastiveLoss,
    add_projection_head,
    create_classifier_v2,
    create_encoder,
)

# Argument
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Which model to warn-up", required=True, type=str)
parser.add_argument("--feature", help="Which feature to use", required=True, type=str)
parser.add_argument(
    "--lr_init", help="Initial Learning Rate", required=True, type=float
)
parser.add_argument("--lr_last", help="Last Learning Rate", required=True, type=float)
parser.add_argument("--unit", help="Unit of Dense Layers", required=True, type=int)
parser.add_argument(
    "--layers", help="Number of layers to not train", required=True, type=float
)
args = parser.parse_args()

# Distribute between GPU and CPU
strategy = tf.distribute.MirroredStrategy()

# initialize the number of epochs to train for and batch size
temperature = 0.05
BS = 128

# load dataset
print("\n[INFO]: Load datase")
all_set = prepare_from_tfrecord(
    tfrecord=join(FEATURES_PATH, f"All_{args.feature}.tfrecords"),
    batch=BS,
    train=True,
    dims=DIMS_MODEL,
)

train_set = prepare_from_tfrecord_v2(
    tfrecord=join(FEATURES_PATH, f"Train_{args.feature}.tfrecords"),
    batch=BS,
    train=True,
    dims=DIMS_MODEL,
)

test_set = prepare_from_tfrecord(
    tfrecord=join(FEATURES_PATH, f"Test_{args.feature}.tfrecords"),
    batch=BS,
    train=False,
    dims=DIMS_MODEL,
)

# ############ TRAIN ENCODER #############

# construct our encoder
print("\n[INFO]: Create encoder")
with strategy.scope():
    resnet = models[args.model](
        include_top=False,
        weights='imagenet',
        input_shape=DIMS_MODEL,
        pooling="avg",
    )

    encoder = create_encoder(
        model=resnet, input_shape=DIMS_MODEL, last_layers=args.layers
    )

# add projection header to model
print("\n[INFO]: Add projection header")
with strategy.scope():
    encoder_with_projection_head = add_projection_head(
        encoder=encoder,
        input_shape=DIMS_MODEL,
        projection_units=args.unit,
    )
    encoder_with_projection_head.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr_init),
        loss=SupervisedContrastiveLoss(temperature),
    )

# train encoder
print("\n[INFO]: Train encoder")
H = encoder_with_projection_head.fit(
    all_set,
    epochs=1,
    steps_per_epoch=98,
)

# ############ TRAIN CLASSIFIER #############

# create and train classifier
print("\n[INFO]: Create and Train classifer")
with strategy.scope():
    classifier = create_classifier_v2(
        encoder,
        input_shape=DIMS_MODEL,
        hidden_units=4 * args.unit,
        num_classes=len(CLASS_NAMES),
        trainable=False,
    )

    classifier.compile(
        optimizer=tf.keras.optimizers.Adam(args.last_ini),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

# callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        join(MODEL_PATH, f"best_{args.feature}_contrastive.h5"),
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
    steps_per_epoch=78,
    validation_steps=9,
)

# show the accuracy on the testing set
(loss, accuracy) = classifier.evaluate(test_set, steps=9)
print(f"[INFO] accuracy after warn up: {accuracy * 100}%")
print(f"[INFO] loss after warn up: {loss}%")
