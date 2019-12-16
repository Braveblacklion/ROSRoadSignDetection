# L. Rozmarynowski; Sentdex(Youtube-Tutorial)

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import math
from tqdm import tqdm
from pathlib import Path

DATADIR = Path('C:/**/pictures/Training/Images')  # Pfad wird für OS automatisch angepasst
training_data = []
IMG_SIZE = 50  # Pixels
M_ROTATION = cv2.getRotationMatrix2D((25, 25), 10, 1)
M_ROTATION_INV = cv2.getRotationMatrix2D((25, 25), -10, 1)
M_TRANSITION = np.float32([[1, 0, 3], [0, 1, 6]])
X = []  # Picture
y = []  # classification Index
PICTURE_GOAL = 2300.0


def data_argumentation(image, nbr):
    # Tilt   = 0, 1,    3,    5,
    # InvT   =                   6, 7, 8,
    # Shift  = 0,    2,    4, 5, 6,    8,
    # Smooth =          3, 4, 5, 6,       9.
    img = image

    if nbr == 0 or nbr == 1 or nbr == 3 or nbr == 5:    # tilt 10%
        img = cv2.warpAffine(img, M_ROTATION, (50, 50))

    if nbr == 6 or nbr == 7 or nbr == 8:                # tilt -10%
        img = cv2.warpAffine(img, M_ROTATION_INV, (50, 50))

    if nbr == 0 or nbr == 2 or nbr == 4 or nbr == 5 or nbr == 6 or nbr == 8:    # shift
        img = cv2.warpAffine(img, M_TRANSITION, (50, 50))

    if nbr == 3 or nbr == 4 or nbr == 5 or nbr == 6 or nbr == 9:    # slight smoothing
        img = cv2.bilateralFilter(img, 9, 20, 20)

    return img


def create_training_data():
    for category in range(0, 43):
        path = os.path.join(DATADIR, str(category))  # create path for Category
        class_num = category   # get classification 0 = STOP, 1 = speedlimit, 2 = ...
        count = 1
        class_records = len(os.listdir(path))
        class_rest = PICTURE_GOAL - class_records
        class_iteration = int(math.ceil(class_rest / class_records))  # divided by class_records round up
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([resized_array, class_num])  # add this to our training_data
                if count < int(class_rest):
                    for t in range(0, class_iteration):
                        change_pic = data_argumentation(resized_array, t)
                        training_data.append([change_pic, class_num])  # add this to our training_data
                        count += 1
            except Exception as e:
                pass
            # except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            # except Exception as e:
            #    print("general exception", e, os.path.join(path,img))


create_training_data()
random.shuffle(training_data)

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

pickle_out = open("X3.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y3.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()