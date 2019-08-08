import numpy as np
import pandas as pd
from optparse import OptionParser
from sys import platform
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PIL

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

parser.add_option("-n", "--ships_number", dest="ships_number", help="Number of ships for detection")
parser.add_option("-s", "--one_ship", dest="one_ship", help="Retreive images with 1 ship only")

(options, args) = parser.parse_args()

path_to_csv = path + "/train_ship_segmentations_v2.csv"
df = pd.read_csv(path_to_csv, index_col=0).dropna()
if options.one_ship:
    print("Using images with only 1 ship")
    df = df.loc[~df.index.duplicated(keep=False)]
print("Number of ships: ", len(df))
if not options.ships_number:
    print("Going with full check")
    number = len(df)
else:
    if int(options.ships_number) > len(df):
        parser.error("Error: Number of ships are greater than total number of ships")
    elif int(options.ships_number) == len(df):
        print("Going with full check anyway")
        number = len(df)
    elif int(options.ships_number) <= 0:
        parser.error("Error: Zero or negative value was provided")
    else:
        number = int(options.ships_number)
        print("Checking randomly selected", number, "ships")

# turn rle example into a list of ints
rle = [int(i) for i in df['EncodedPixels']['55bd28f41.jpg'].split()]
# turn list of ints into a list of (`start`, `length`) `pairs`
pairs = list(zip(rle[0:-1:2], rle[1::2]))

start = pairs[0][0]

coordinate = (start % 768, start // 768)

back = 768 * coordinate[1] + coordinate[0]

pixels = [(pixel_position % 768, pixel_position // 768)
          for start, length in pairs
          for pixel_position in range(start, start + length)]


def checker(rle_code):
    rle_code = [int(i) for i in rle_code.split()]
    rle_pixels = [int(i) for i in (rle_code[0::2])]
    X_pixels = [s % 768 for s in rle_pixels]
    Y_pixels = [s // 768 for s in rle_pixels]
    a = min(X_pixels)
    b = max(X_pixels)
    c = abs(b - a)
    y1 = [i for i, x in enumerate(X_pixels) if x == a]
    y2 = [i for i, x in enumerate(X_pixels) if x == b]
    d = abs(Y_pixels[y2[0]] - Y_pixels[y1[0]])
    try:
        angle = d / c
    except ZeroDivisionError:
        return True
    if angle < 0.2 or angle > 5.1:
        return True
    else:
        return False


def rle_to_pixels(rle_code):
    '''
    Transforms a RLE code string into a list of pixels of a (768, 768) canvas
    '''
    rle_code = [int(i) for i in rle_code.split()]
    pixels = [(pixel_position % 768, pixel_position // 768)
              for start, length in list(zip(rle_code[0:-1:2], rle_code[1::2]))
              for pixel_position in range(start, start + length)]
    return pixels


# An image may have more than one row in the df,
# Meaning that the image has more than one ship present
# This part resets index for next ship occurrence
df = df.reset_index()
# Counter for ships detected and set incorrectly by rle_to_pixels, TODO: check why
incorrect = 0
correct = 0
notAllignedships = 0
used_rows = []

for i in tqdm(range(number)):
    if number == len(df):
        row_index = i  # take all rows one-by-one
    else:
        while True:
            row_index = np.random.randint(len(df))  # take a random row from the df
            if row_index not in used_rows:
                used_rows.append(row_index)
                break
    if checker(df.loc[row_index, 'EncodedPixels']):
        mask_pixels = rle_to_pixels(df.loc[row_index, 'EncodedPixels'])
        tuple_y, tuple_x = zip(*mask_pixels)
        table_x = list(tuple_x)
        table_y = list(tuple_y)
        x_min = min(table_x)
        y_min = min(table_y)
        x_max = max(table_x)
        y_max = max(table_y)
        cond1 = x_min == 0 and x_max == 767
        cond2 = x_max == 0 and x_min == 767
        cond3 = y_min == 0 and y_max == 767
        cond4 = y_max == 0 and y_min == 767

        # decoder might makes mistakes - make sure that there's no BB for whole width/height
        if cond1 or cond2 or cond3 or cond4:
            incorrect += 1
        # Following code is not necessary right now
        # with open("incorrect_images.csv", "a") as g:
        #     g.write("input/train_v2/")
        #     print(df.loc[row_index, 'ImageId'], x_min, y_min, x_max, y_max, "ship", sep=',',
        #           file=g)

        else:
            # NOTE: uncomment following part for checking if the Bounding Boxes are correctly selected
            # load_img = lambda filename: np.array(PIL.Image.open(f"./input/train_v2/{filename}"))
            # im = np.array(load_img(df.loc[row_index, 'ImageId']), dtype=np.uint8)
            # # Create figure and axes
            # fig, ax = plt.subplots(1)
            # # Display the image
            # ax.imshow(im)
            # # Create a Rectangle patch
            # rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, linewidth=1,
            #                          edgecolor='r', facecolor='none')
            # # Add the patch to the Axes
            # ax.add_patch(rect)
            # ax.set_title(df.loc[row_index, 'ImageId'])
            # plt.show()

            correct += 1
            if platform == "linux" or platform == "linux2":
                with open("entry_data.csv", "a") as f:
                    f.write("/content/pracaMgr/input/train_v2/")
                    print(df.loc[row_index, 'ImageId'], x_min, y_min, x_max, y_max, "ship", sep=',',
                          file=f)
            else:
                with open("entry_data.csv", "a") as f:
                    f.write("input/train_v2/")
                    print(df.loc[row_index, 'ImageId'], x_min, y_min, x_max, y_max, "ship", sep=',',
                          file=f)
    else:
        notAllignedships += 1

print("Checked ships:", number)
print("Incorrect ships:", incorrect)
print("Correct ships:", correct)
print("Not alligned ships:", notAllignedships)
print("Finished")
