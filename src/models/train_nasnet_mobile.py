# -*- coding: utf-8 -*-
"""
    Script used to train nasnet model
"""
from os.path import join

import tensorflow as tf

from src.config import CLASS_NAMES, FEATURES_PATH, LOGS_PATH, MODEL_PATH
from src.data import DatasetGenerator, load_dataset_from_generator, prepare

# ############ WARM UP #############

# initialize the number of epochs to train for and batch size
LR = 1e-5
BS = 32
# data generator: train
train_generator = DatasetGenerator(
    db_path=join(FEATURES_PATH, "train_set.hdf5"),
    binarize=True,
    classes=len(CLASS_NAMES),
)
train_set = load_dataset_from_generator(
    generator=train_generator, batch=BS, shape=224, train=True
)

# data generator: test
test_generator = DatasetGenerator(
    db_path=join(FEATURES_PATH, "test_set.hdf5"),
    binarize=True,
    classes=len(CLASS_NAMES),
)
test_set = load_dataset_from_generator(
    generator=test_generator, batch=BS, shape=224, train=False
)

# construct our model
print("[INFO]: Create model")
head = tf.keras.applications.NASNetMobile(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
    classes=len(CLASS_NAMES),
)
# Freeze the pretrained weights
head.trainable = False

# Rebuild top
x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(head.output)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2, name="top_dropout")(x)
outputs = tf.keras.layers.Dense(9, activation="sigmoid")(x)

# Compile
model = tf.keras.Model(head.input, outputs, name="NASNetMobile")
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["loss"])

# callbacks
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=join(LOGS_PATH, "nasnet"), profile_batch=0)
]

# train the head of the network
print("[INFO] training: warm up ...")
H = model.fit(train_set, validation_data=test_set, epochs=15, callbacks=callbacks)

# ############ FINE TURN #############

# unfreeze layers
for layer in model.layers[:-4]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["loss"])

# callbacks
callbacks = [
    tf.keras.callbacks.TensorBoard(
        log_dir=join(LOGS_PATH, "nasnet-fine"), profile_batch=0
    ),
    tf.keras.callbacks.ModelCheckpoint(
        join(MODEL_PATH, "best_nasnet.h5"),
        monitor="val_loss",
        mode="min",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
    ),
]

# train the head of the network
print("[INFO] training: fine tune...")
H = model.fit(train_set, validation_data=test_set, epochs=25, callbacks=callbacks)

# close
train_generator.close()
test_generator.close()
