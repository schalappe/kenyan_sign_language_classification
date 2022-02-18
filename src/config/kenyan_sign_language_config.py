# -*- coding: utf-8 -*-
"""
    Contains all necessary paths
"""
from os.path import abspath, dirname, join

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))

DATA_PATH = join(ROOT_PATH, "data")
MODEL_PATH = join(ROOT_PATH, "models")

IMAGES_PATH = join(DATA_PATH, "Images")
SUBMIT_PATH = join(DATA_PATH, "submit")
FEATURES_PATH = join(DATA_PATH, "features")

INPUT_CSV = join(DATA_PATH, "Train.csv")
OUTPUT_CSV = join(DATA_PATH, "Test.csv")

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
DIMS_MODEL = (224, 224, 3)
DIMS_MODEL_LARGE = (331, 331, 3)
