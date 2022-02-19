# -*- coding: utf-8 -*-
"""
    Script used to fine-tune a model
"""
import argparse
from os.path import join

import tensorflow as tf

from addons import models
from src.config import CLASS_NAMES, DIMS_MODEL, FEATURES_PATH, MODEL_PATH
from src.data import prepare_from_tfrecord, prepare_from_tfrecord_v2
from src.models import FCHeadNetV2

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

# ############ WARM UP #############

# Distribute between GPU and CPU
strategy = tf.distribute.MirroredStrategy()


# initialize the number of epochs to train
BS = 32
epochs_init = 20
epochs_last = 30

# load dataset
print("\n[INFO]: Load datase")
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

# construct our model
print("\n[INFO]: Create model")
with strategy.scope():
    head = models[args.model](
        input_shape=DIMS_MODEL,
        include_top=False,
        weights="imagenet",
    )

    # Freeze the pretrained weights
    head.trainable = False

    # Rebuild top
    outputs = FCHeadNetV2.build(
        base_model=head, len_class=len(CLASS_NAMES), dense_unit=args.unit
    )

    # Compile
    model = tf.keras.Model(head.input, outputs)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.lr_init,  decay=args.lr_init / epochs_init
    )
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

# train the head of the network
print("\n[INFO] training: warm up ...")
H = model.fit(
    train_set,
    validation_data=test_set,
    epochs=epochs_init,
    steps_per_epoch=314,
    validation_steps=39,
)

# show the accuracy on the testing set
(loss, accuracy) = model.evaluate(test_set, steps=39)
print(f"[INFO] accuracy after warn up: {accuracy * 100}%")
print(f"[INFO] loss after warn up: {loss}%")

# ############ FINE TURN #############

# unfreeze layers
last_layers = int(args.layers * len(model.layers))
if last_layers == 0:
    model.trainable = True
else:
    for layer in model.layers[last_layers:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr_last)
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"],
)

# callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
    tf.keras.callbacks.ModelCheckpoint(
        join(MODEL_PATH, f"best_{args.model}.h5"),
        monitor="val_loss",
        mode="min",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
    ),
]

# train the head of the network
print("\n[INFO] training: fine tune...")
H = model.fit(
    train_set,
    validation_data=test_set,
    epochs=epochs_init+epochs_last,
    callbacks=callbacks,
    initial_epoch=H.epoch[-1],
    steps_per_epoch=314,
    validation_steps=39,
)

# show the accuracy on the testing set
(loss, accuracy) = model.evaluate(test_set, steps=39)
print(f"[INFO] accuracy after fine-tune: {accuracy * 100}%")
print(f"[INFO] loss after warn up: {loss}%")
