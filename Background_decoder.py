import numpy as np
import pandas as pd
from optparse import OptionParser
from sys import platform
from tqdm import tqdm
import random

if platform == "linux" or platform == "linux2":
    from IPython.core.display import display
    path = str("/content/pracaMgr/input")
elif platform == "darwin":
    path = str("./input")
elif platform == "win32":
    path = str("../input")
else:
    print("Undefined platform ", platform)
    NameError("Error: System type not defined. Please check platform name")

parser = OptionParser()

parser.add_option("-n", "--images_number", dest="images_number", help="Number of images for background detection")

(options, args) = parser.parse_args()

path_to_csv = path + "/train_ship_segmentations_v2.csv"
df = pd.read_csv(path_to_csv, index_col=0)
df = df[df['EncodedPixels'].isna()]
print("Number of images without ships:", len(df))

if not options.images_number:
    number = len(df)
    print("Going with full check with", number, "images")
else:
    if int(options.images_number) > len(df):
        parser.error("Error: Number of images are greater than total number of images, "
                     "total amount images is " + str(len(df)))
    elif int(options.images_number) == len(df):
        print("Going with full check anyway")
        number = len(df)
    elif int(options.images_number) <= 0:
        parser.error("Error: Zero or negative value was provided")
    else:
        number = int(options.images_number)
        print("Checking randomly selected", number, "images")

used_rows = []
df = df.reset_index()

for i in tqdm(range(number)):
    if number == len(df):
        row_index = i  # take all rows one-by-one
    else:
        while True:
            row_index = np.random.randint(len(df))  # take a random row from the df
            if row_index not in used_rows:
                used_rows.append(row_index)
                break

    size = random.randrange(1, 10)
    x_min = random.randrange(2, 500)
    x_max = x_min + 20 * size
    y_min = x_min + 15 * size
    size = random.randrange(1, 10)
    y_max = y_min + 15 * size
    if x_max > 767:
        x_max = 767
    if y_max > 767:
        y_max = 767

    if platform == "linux" or platform == "linux2":
        with open("entry_data.csv", "a") as f:
            f.write("/content/pracaMgr/input/train_v2/")
            print(df.loc[row_index, 'ImageId'], x_min, y_min, x_max, y_max, "bg", sep=',', file=f)
    else:
        with open("entry_data.csv", "a") as f:
            f.write("input/train_v2/")
            print(df.loc[row_index, 'ImageId'], x_min, y_min, x_max, y_max, "bg", sep=',', file=f)

print("Checked images:", number)
print("Finished")
