# -*- coding: utf-8 -*-
"""
    Script used to train resnet model
"""
from os.path import join

import tensorflow as tf
from head_net import NormHeadNet

from src.config import CLASS_NAMES, FEATURES_PATH, LOGS_PATH, MODEL_PATH
from src.data import prepare_from_tfrecord

# ############ WARM UP #############

# initialize the number of epochs to train for and batch size
LR_init = 1e-3
LR_last = 1e-3
BS = 32

# load dataset
print("\n[INFO]: Load datase")
train_set = prepare_from_tfrecord(
    tfrecord=join(FEATURES_PATH, "Train.tfrecords"), batch=BS, train=True
)

test_set = prepare_from_tfrecord(
    tfrecord=join(FEATURES_PATH, "Test.tfrecords"), batch=BS, train=False
)

# construct our model
print("\n[INFO]: Create model")
head = tf.keras.applications.ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
    classes=len(CLASS_NAMES),
)

# Freeze the pretrained weights
head.trainable = False

# Rebuild top
outputs = NormHeadNet.build(base_model=head, len_class=len(CLASS_NAMES))

# Compile
model = tf.keras.Model(head.input, outputs, name="ResNet50")
optimizer = tf.keras.optimizers.RMSprop(learning_rate=LR_init)
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"],
)

# callbacks
callbacks = [
    tf.keras.callbacks.TensorBoard(
        log_dir=join(LOGS_PATH, "resnet-norm"), profile_batch=0
    )
]

# train the head of the network
print("\n[INFO] training: warm up ...")
H = model.fit(train_set, validation_data=test_set, epochs=15, callbacks=callbacks)

# ############ FINE TURN #############

# unfreeze layers
for layer in model.layers[:-4]:
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
    tf.keras.callbacks.TensorBoard(
        log_dir=join(LOGS_PATH, "resnet-fine-norm"), profile_batch=0
    ),
    tf.keras.callbacks.ModelCheckpoint(
        join(MODEL_PATH, "best_resnet_norm.h5"),
        monitor="val_loss",
        mode="min",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
    ),
]

# train the head of the network
print("\n[INFO] training: fine tune...")
H = model.fit(train_set, validation_data=test_set, epochs=25, callbacks=callbacks)

# show the accuracy on the testing set
(loss, accuracy) = model.evaluate(test_set)
print(f"[INFO] accuracy: {accuracy * 100}%")
