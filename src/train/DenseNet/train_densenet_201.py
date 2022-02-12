# -*- coding: utf-8 -*-
"""
    Script used to train DenseNet 201 model
"""
from os.path import join

import tensorflow as tf

from src.config import CLASS_NAMES, DIMS_MODEL, FEATURES_PATH, MODEL_PATH
from src.data import prepare_from_tfrecord
from src.models import NormHeadNetV2

# ############ WARM UP #############

# initialize the number of epochs to train for and batch size
LR_init = 1e-4
LR_last = 1e-4
BS = 32

# load dataset
print("\n[INFO]: Load datase")
train_set = prepare_from_tfrecord(
    tfrecord=join(FEATURES_PATH, "Train_DenseNet.tfrecords"), batch=BS, train=True
)

test_set = prepare_from_tfrecord(
    tfrecord=join(FEATURES_PATH, "Test_DenseNet.tfrecords"), batch=BS, train=False
)

# construct our model
print("\n[INFO]: Create model")
head = tf.keras.applications.DenseNet201(
    input_shape=DIMS_MODEL,
    include_top=False,
    weights="imagenet",
)

# Freeze the pretrained weights
head.trainable = False

# Rebuild top
outputs = NormHeadNetV2.build(
    base_model=head,
    len_class=len(CLASS_NAMES),
    dense_unit=4086
)
print(len(head.layers))

# Compile
model = tf.keras.Model(head.input, outputs, name="DenseNet201")
optimizer = tf.keras.optimizers.Adam(learning_rate=LR_init, decay=LR_init / 20)
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
    epochs=20,
    steps_per_epoch=157,
    validation_steps=39,
)

# show the accuracy on the testing set
(loss, accuracy) = model.evaluate(test_set, steps=39)
print(f"[INFO] accuracy after warn up: {accuracy * 100}%")
print(f"[INFO] loss after warn up: {loss}%")

# ############ FINE TURN #############

# unfreeze layers
for layer in model.layers[:500]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

optimizer = tf.keras.optimizers.SGD(learning_rate=LR_last)
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"],
)

# callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
    tf.keras.callbacks.ModelCheckpoint(
        join(MODEL_PATH, "best_densenet_201_2.h5"),
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
    epochs=50,
    callbacks=callbacks,
    initial_epoch=H.epoch[-1],
    steps_per_epoch=157,
    validation_steps=39,
)

# show the accuracy on the testing set
(loss, accuracy) = model.evaluate(test_set, steps=39)
print(f"[INFO] accuracy after fine-tune: {accuracy * 100}%")
print(f"[INFO] loss after warn up: {loss}%")
