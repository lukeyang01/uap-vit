import csv
import os
import shutil
from os.path import join


TRAINING_PATH = "imagenet_data/ILSVRC/Data/CLS-LOC/train"
VALIDATION_PATH = "imagenet_data/ILSVRC/Data/CLS-LOC/val"
TARGETS = "imagenet_data/LOC_val_solution.csv"


for dir in os.listdir(TRAINING_PATH):
    os.makedirs(join(VALIDATION_PATH, dir), exist_ok=True)


with open(TARGETS, encoding="utf-8") as infile:
    reader = csv.reader(infile)
    next(reader)

    for image_id, prediction_string in reader:
        label = prediction_string.split()[0]
        src = f"{VALIDATION_PATH}/{image_id}.JPEG"
        dst = f"{VALIDATION_PATH}/{label}/{image_id}.JPEG"
        try:
            shutil.move(src, dst)
        except:
            print(f"Error Moving {src} to {dst}")
