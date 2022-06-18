import os
from os.path import join
import csv
import cv2

translate = {'b': 0, 'g': 1, 'r': 2}


def img_merge(files, input_dir, output_dir, output_name):

    img = [
        cv2.imread(join(input_dir, files[i]), cv2.IMREAD_GRAYSCALE)
        for i in range(3)
    ]

    image_merge = cv2.merge(img)

    return cv2.imwrite(join(output_dir, output_name), image_merge)


#os.path.join(os.path.dirname(__file__),)
def merge_channels(input_dir, output_dir):
    images = {}
    with open(join(input_dir, 'description.csv'), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            i, c, path = int(
                row['full_image_index']), row['color'], row['image_path']
            if i not in images:
                images[i] = [''] * 3
            images[i][translate[c]] = path

    c = 0
    with open(join(input_dir, 'image_counter.txt'), 'r') as f:
        c = int(f.readline().strip())
    assert len(images) == c

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in images:
        img_merge(images[i], join(input_dir, 'data'), output_dir, f'{i:05}.jpg')
