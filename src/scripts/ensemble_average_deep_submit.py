# -*- coding: utf-8 -*-
"""
    Script used to create a submit file from ensemble learning
"""
import argparse
import csv
from os.path import join, sep

import numpy as np
import tensorflow as tf
from pandas import read_csv
from tqdm import tqdm

from src.addons import GCAdam
from src.config import (
    CLASS_NAMES,
    DIMS_MODEL,
    IMAGES_PATH,
    MODEL_PATH,
    OUTPUT_CSV,
    SUBMIT_PATH,
)
from src.processors import load_image, model_preprocess

# load model and store preprocessor
selected = [
    ("EfficientNetB7", "EfficientNet"),
    ("DenseNet201", "DenseNet"),
    ("ResNet152V2", "ResNet"),
]
models = []
print("[INFO]: loading model ...")
for candidate in tqdm(selected, ncols=100, desc="Load model ...:", colour="red"):
    models.append(
        {
            "model": tf.keras.models.load_model(
                join(MODEL_PATH, f"best_{candidate[0]}.h5"),
                custom_objects={"GCAdam": GCAdam},
            ),
            "preprocessor": model_preprocess[candidate[1]],
        }
    )

# load dataset
print("[INFO]: Load dataset")
submit_set = read_csv(OUTPUT_CSV, header="infer")
submit_set = [join(IMAGES_PATH, image + ".jpg") for image in submit_set.img_IDS]


# create submit file
submit_csv = join(SUBMIT_PATH, "sub-average-best-model.csv")
with open(submit_csv, "w") as submit:
    writer = csv.writer(submit)
    writer.writerow(["img_IDS"] + CLASS_NAMES)


# process on submit image
for path in tqdm(submit_set, ncols=100, desc="Prediction ..."):
    predictions = []
    for candidate in models:
        # load and preprocess image
        image = load_image(path, preprocess_input=candidate["preprocessor"])
        image = tf.image.resize(image, [DIMS_MODEL[0], DIMS_MODEL[1]])
        image = tf.expand_dims(image, axis=0)

        # prediction
        predictions.append(candidate["model"].predict(image)[0])

    # write on submit file
    preds = np.average(predictions, axis=0)
    with open(submit_csv, "a+", newline="") as submit:
        writer = csv.writer(submit)
        writer.writerow(
            [str(path.split(sep)[-1].split(".")[0])] + [str(pred) for pred in preds]
        )

print("[INFO]: Prediction complete")
