import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from matplotlib.colors import hsv_to_rgb


def calc_metric(image, x, y, w, h, k=4):
    img = image[y:y + h, x:x + w]
    img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    clt = KMeans(n_clusters=k)
    labels = clt.fit_predict(img)
    label_counts = Counter(labels)
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
    return list(
        dominant_color)  # HSV h(0..180) s(0..255) v(0..255) (opencv-like)


# mouse callback function
def set_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img, image
    if event == cv2.EVENT_LBUTTONDOWN:
        img = image.copy()
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        color = calc_metric(image, min(ix, x), min(iy, y), abs(ix - x),
                            abs(iy - y))
        print(color)
        color = np.array(color) / np.array([180, 255, 255])
        r, g, b = hsv_to_rgb(color) * 255
        print(color)
        cv2.rectangle(img, (ix, iy), (x, y), (b, g, r), -1)


def debug(file):
    global ix, iy, drawing, img, image
    drawing = False
    ix, iy = -1, -1
    image = cv2.imread(file)
    img = image.copy()
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', set_rectangle)
    while (1):
        cv2.imshow('image', img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()