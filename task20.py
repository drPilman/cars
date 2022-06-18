from os import listdir
from os.path import isfile, join
import csv
import tempfile
import shutil
import torch
import pandas as pd
from task10 import merge_channels

model = torch.hub.load('ultralytics/yolov5', 'yolov5l', _verbose=False)

CONF_THRESH = 0.35


def find_car(input_dir, output_cars="output.csv"):
    dirpath = tempfile.mkdtemp()
    merge_channels(input_dir, dirpath)
    with open(output_cars, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile,
                            delimiter=',',
                            quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)
        for filename in listdir(dirpath):
            path = join(dirpath, filename)
            if isfile(path):
                results = model(path)
                df = results.pandas().xyxy[0]
                df_filtered = df[(df['class'] == 2) | (df['class'] == 7) |
                                 (df['class'] == 5)]  # only car track and bus
                flag = df_filtered['confidence'].max() >= CONF_THRESH
                writer.writerow([filename, 'TRUE' if flag else 'FALSE'])

    shutil.rmtree(dirpath)
