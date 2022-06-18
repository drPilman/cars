import csv
import torch
import pandas as pd
from os.path import join


def load(file):
    r = set()
    with open(file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='\n')
        for name, val in reader:
            if val.lower() == 'true':
                r.add(name)
    return r


m = load('output.csv')

n = load('data/1/val.csv')
a = n.symmetric_difference(m)
print(a, m)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', _verbose=False)

for file in a:

    results = model(join('data/2', file))
    results.show()
    df = results.pandas().xyxy[0]
    print(df)
