# -*- coding: utf-8 -*-
"""
    Script used to create a submit file
"""
import argparse
import csv
from os.path import join, sep

import tensorflow as tf
from pandas import read_csv
from tqdm import tqdm

from addons import model_preprocess
from src.config import (
    CLASS_NAMES,
    DIMS_MODEL,
    IMAGES_PATH,
    MODEL_PATH,
    OUTPUT_CSV,
    SUBMIT_PATH,
)
from src.processors import load_image

# Argument
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Which model to warn-up", required=True, type=str)
parser.add_argument(
    "--preprocess", help="Which feature to use", required=True, type=str
)
args = parser.parse_args()

print("[INFO]: Load dataset")
submit_set = read_csv(OUTPUT_CSV, header="infer")
submit_set = [join(IMAGES_PATH, image + ".jpg") for image in submit_set.img_IDS]

print("[INFO]: loading model ...")
model = tf.keras.models.load_model(join(MODEL_PATH, f"best_{args.model}.h5"))

# create submit file
submit_csv = join(SUBMIT_PATH, f"sub-{args.model}-best.csv")
with open(submit_csv, "w") as submit:
    writer = csv.writer(submit)
    writer.writerow(["img_IDS"] + CLASS_NAMES)

# process on submit image
for path in tqdm(submit_set, ncols=100, desc="Prediction ..."):
    # load and preprocess image
    image = load_image(path, preprocess_input=model_preprocess[args.preprocess])
    image = tf.image.resize(image, [DIMS_MODEL[0], DIMS_MODEL[1]])
    image = tf.expand_dims(image, axis=0)

    # prediction
    preds = model.predict(image)[0]

    # write on submit file
    with open(submit_csv, "a+", newline="") as submit:
        writer = csv.writer(submit)
        writer.writerow(
            [str(path.split(sep)[-1].split(".")[0])] + [str(pred) for pred in preds]
        )

print("[INFO]: Prediction complete")
