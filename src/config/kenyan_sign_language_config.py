# -*- coding: utf-8 -*-
"""
    Contains all necessary paths
"""
from os.path import abspath, dirname, join

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))

DATA_PATH = join(ROOT_PATH, "data")
MODEL_PATH = join(ROOT_PATH, "models")

IMAGES_PATH = join(DATA_PATH, "Images")
TRAIN_PATH = join(DATA_PATH, "train")
SUBMIT_PATH = join(DATA_PATH, "submit")
LOGS_PATH = join(DATA_PATH, "logs")
FEATURES_PATH = join(DATA_PATH, "features")

INPUT_CSV = join(DATA_PATH, "Train.csv")

CLASS_NAMES = [
    "Church",
    "Enough/Satisfied",
    "Friend",
    "Love",
    "Me",
    "Mosque",
    "Seat",
    "Temple",
    "You",
]

DIMS_IMAGE = (256, 256, 3)
