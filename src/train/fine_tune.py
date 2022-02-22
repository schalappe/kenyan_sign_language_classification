# -*- coding: utf-8 -*-
"""
    Script used to fine-tune a model
"""
import argparse
from os.path import join

import tensorflow as tf

from src.addons import GCSGD, GCAdam
from src.config import CLASS_NAMES, DIMS_MODEL, MODEL_PATH
from src.data import return_dataset
from src.models import FineTuneModel

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
train_set, test_set = return_dataset(family_model=args.feature, batch_size=BS)

# construct our model
print("\n[INFO]: Create model")
with strategy.scope():
    model = FineTuneModel.build(
        model_name=args.model,
        dims=DIMS_MODEL,
        num_class=len(CLASS_NAMES),
        hidden_unit=args.unit,
    )

    # Compile
    optimizer = GCAdam(learning_rate=args.lr_init, decay=args.lr_init / epochs_init)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

# train the head of the network
print("\n[INFO] training: warm up ...")
H_init = model.fit(
    train_set,
    validation_data=test_set,
    epochs=epochs_init,
    steps_per_epoch=156,
    validation_steps=39,
)

# show the accuracy on the testing set
(loss, accuracy) = model.evaluate(test_set, steps=39)
print(f"[INFO] accuracy after warn up: {accuracy * 100}%")
print(f"[INFO] loss after warn up: {loss}%")

# ############ FINE TURN #############

# unfreeze layers
last_layers = int(args.layers * len(model.layers))
for layer in model.layers[last_layers:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

optimizer = GCAdam(learning_rate=args.lr_last)
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"],
)

# callbacks
callbacks = [
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
model.fit(
    train_set,
    validation_data=test_set,
    epochs=epochs_init + epochs_last,
    callbacks=callbacks,
    initial_epoch=H_init.epoch[-1],
    steps_per_epoch=156,
    validation_steps=39,
)

# show the accuracy on the testing set
(loss, accuracy) = model.evaluate(test_set, steps=39)
print(f"[INFO] accuracy after fine-tune: {accuracy * 100}%")
print(f"[INFO] loss after warn up: {loss}%")
