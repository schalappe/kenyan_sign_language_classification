# -*- coding: utf-8 -*-
"""
    Script used to search the best parameters for Dense Net
"""
import argparse
from os.path import join

import tensorflow as tf
from numpy.random import RandomState
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import ParameterSampler

from addons import models
from src.config import CLASS_NAMES, DIMS_MODEL, FEATURES_PATH
from src.data import prepare_from_tfrecord
from src.models import NormHeadNetV2

# Argument
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Which model to warn-up", required=True)
parser.add_argument("--feature", help="Which feature to use", required=True)
args = parser.parse_args()

# HyperParameters
params_grid = {
    "LR_init": [1e-2, 1e-3, 1e-4],
    "unit_denses": [128 * 2**i for i in range(4)],
    "best_layers": [0.5, 0.75],
}
rng = RandomState(42)
params_list = list(ParameterSampler(params_grid, n_iter=8, random_state=rng))

# parameters
BS = 32

# load dataset
print("\n[INFO]: Load datase")
train_set = prepare_from_tfrecord(
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

# Table
table = Table(title="Parameters")
table.add_column("N°")
table.add_column("LR_init")
table.add_column("unit_denses")
table.add_column("best_layers")

# results
results = []

for index, params in enumerate(params_list):
    print(f"# ############ PARAMS: {index+1} #############")
    tf.keras.backend.clear_session()

    # print log
    table.add_row(
        str(index + 1),
        str(params["LR_init"]),
        str(params["unit_denses"]),
        str(params["best_layers"]),
    )
    console = Console()
    console.print(table)

    # ############ WARM UP #############

    # construct our model
    print("\n[INFO]: Create model")
    head = models[args.model](
        input_shape=DIMS_MODEL,
        include_top=False,
        weights="imagenet",
    )

    # Freeze the pretrained weights
    head.trainable = False

    # Rebuild top
    outputs = NormHeadNetV2.build(
        base_model=head, len_class=len(CLASS_NAMES), dense_unit=params["unit_denses"]
    )

    # Compile
    model = tf.keras.Model(head.input, outputs)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=params["LR_init"], decay=params["LR_init"] / 20
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
        epochs=20,
        steps_per_epoch=314,
        validation_steps=39,
        verbose=2,
    )

    # ############ FINE TURN #############

    # unfreeze layers
    last_layers = int(params["best_layers"] * len(model.layers))
    for layer in model.layers[:last_layers]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

    # callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8),
    ]

    # train the head of the network
    print("\n[INFO] training: fine tune...")
    H = model.fit(
        train_set,
        validation_data=test_set,
        epochs=50,
        initial_epoch=H.epoch[-1],
        callbacks=callbacks,
        steps_per_epoch=314,
        validation_steps=39,
        verbose=2,
    )

    # show the accuracy on the testing set
    (loss, accuracy) = model.evaluate(test_set, steps=39)

    # store result
    results.append(
        (
            params["LR_init"],
            params["unit_denses"],
            params["best_layers"],
            accuracy,
            loss,
        )
    )

# ############ BILAN #############
results.sort(key=lambda x: x[-1])
bilan = Table(title="Bilan for fine tune")
table.add_column("N°")
table.add_column("LR_init")
table.add_column("unit_denses")
table.add_column("best_layers")
table.add_column("accuracy")
table.add_column("loss")

for index, result in enumerate(results):
    bilan.add_row(
        str(index + 1),
        str(result[0]),
        str(result[1]),
        str(result[2]),
        str(result[3]),
        str(result[4]),
    )

console = Console()
console.print(table)
