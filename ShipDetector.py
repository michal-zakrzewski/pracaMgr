import numpy as np  # linear algebra
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
#import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import kaggle
import keras_preprocessing
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
import os

print(tf.__version__)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# save filepath to variable for easier access
ship_path_files = '../input/train_ship_segmentations/sample_submission_v2.csv'
# read the data and store data in DataFrame titled ship_data
ship_data = pd.read_csv(ship_path_files)
# print a summary of the data in SHIP data
ship_data.describe()
# path to training directory
train_images_dir = '../input/train'
# path to test directory
test_images_dir = '../input/test'
#path to masks
masks_overlayed_dir = 'output/masks'

print(os.listdir("../input"))
train = os.listdir('../input/train')
print(len(train))

test = os.listdir('../input/test')
print(len(test))

BACKBONE = 'resnet34'
preprocess_input = get_preprocessing(BACKBONE)

model = Unet()

# load your data
x_train, y_train, x_val, y_val = load_data(train_images_dir)

# preprocess input
x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)

# define model
model = Unet(BACKBONE, encoder_weights='imagenet')
model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])

# fit model
model.fit(
    x=x_train,
    y=y_train,
    batch_size=16,
    epochs=100,
    validation_data=(x_val, y_val),
)