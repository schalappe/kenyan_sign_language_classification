# -*- coding: utf-8 -*-
"""
    Script used to search the best parameters for Dense Net
"""
import argparse

import tensorflow as tf
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import ParameterGrid

from src.addons import GCSGD, GCAdam, GCRMSprop
from src.config import CLASS_NAMES, DIMS_MODEL
from src.data import return_dataset
from src.models import FineTuneModel

# Argument
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Which model to warn-up", required=True)
parser.add_argument("--feature", help="Which feature to use", required=True)
args = parser.parse_args()

# HyperParameters
optimizers = {
    "Adam": GCAdam,
    "RMSprop": GCRMSprop,
    "SGD": GCSGD,
}
params_grid = {
    "LR_init": [1e-2, 1e-3, 1e-4],
    "unit_denses": [128 * 2**i for i in range(3)],
    "optimizer": optimizers.keys(),
}

params_list = list(ParameterGrid(params_grid))

# parameters
BS = 32
epochs_init = 20

# load dataset
print("\n[INFO]: Load datase")
train_set, test_set = return_dataset(family_model=args.feature, batch_size=BS)

# Table
table = Table(title="Parameters")
table.add_column("N°")
table.add_column("LR_init")
table.add_column("unit_denses")
table.add_column("optimizer")

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
        str(params["optimizer"]),
    )
    console = Console()
    console.print(table)

    # ############ WARM UP #############

    # construct our model
    print("\n[INFO]: Create model")
    model = FineTuneModel.build(
        model_name=args.model,
        dims=DIMS_MODEL,
        num_class=len(CLASS_NAMES),
        hidden_unit=params["unit_denses"],
    )

    # Compile
    optimizer = optimizers[params["optimizer"]](
        learning_rate=params["LR_init"], decay=params["LR_init"] / epochs_init
    )
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

    # train the head of the network
    print("\n[INFO] training: warm up ...")
    model.fit(
        train_set,
        validation_data=test_set,
        epochs=epochs_init,
        steps_per_epoch=156,
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
            params["optimizer"],
            accuracy,
            loss,
        )
    )

# ############ BILAN #############
results.sort(key=lambda x: x[-1])
bilan = Table(title="Bilan for fine tune")
bilan.add_column("N°")
bilan.add_column("LR_init")
bilan.add_column("unit_denses")
bilan.add_column("optimizer")
bilan.add_column("accuracy")
bilan.add_column("loss")

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
console.print(bilan)
