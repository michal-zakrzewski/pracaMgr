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

(options, args) = parser.parse_args()

path_to_csv = path + "/train_ship_segmentations_v2.csv"
df = pd.read_csv(path_to_csv, sep='\s*,\s*', engine = 'python', index_col=0).dropna()

ids = df.groupby('ImageId').aggregate('count')
ids = ids[ids.EncodedPixels >= 3].index.tolist()
df = df[df['ImageId'].isin(ids)]
df.set_index('ImageId', inplace=True)
print(df)

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

# zamien string na liste intow
rle = [int(i) for i in df['EncodedPixels']['55bd28f41.jpg'].split()]
# zamien liste liczb na liste par (poczatek, koniec)
pairs = list(zip(rle[0:-1:2], rle[1::2]))

start = pairs[0][0]

coordinate = (start % 768, start // 768)

back = 768 * coordinate[1] + coordinate[0]

pixels = [(pixel_position % 768, pixel_position // 768)
          for start, length in pairs
          for pixel_position in range(start, start + length)]


def rle_to_pixels(rle_code):
    """
    Zmiana listy pikseli zakodowana w RLE na wspolrzedne (x,y) kazdego piksela
    """
    rle_code = [int(i) for i in rle_code.split()]
    pixels = [(pixel_position % 768, pixel_position // 768)
                 for start, length in list(zip(rle_code[0:-1:2], rle_code[1::2]))
                 for pixel_position in range(start, start + length)]
    return pixels

# Jesli zdjecie ma wiecej niz 1 wpis w plik, to jest wiecej niz 1 statek na zdjeciu
# Trzeba zresetowac indeks przed kolejnym statkiem
df = df.reset_index()
# Policz statki, ktore przejda przez ten dekoder poprawnie
incorrect = 0
correct = 0
used_rows = []

for i in tqdm(range(number)):
    if number == len(df):
        row_index = i  # uzyj wszystkich statkow jeden po drugim
    else:
        while True:
            row_index = np.random.randint(len(df))  # wylosuj wiersz w df
            if row_index not in used_rows:
                used_rows.append(row_index)
                break
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

    # upewnij sie, ze nie ma statkow zajmujacych caly ekran (blad dekodera)
    if cond1 or cond2 or cond3 or cond4:
        incorrect += 1

    else:
        # =====Ten fragment kodu mozna odkomentowac celem sprawdzenia, czy statek jest oznaczany poprawnie

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

print("Checked ships:", number)
print("Incorrect ships:", incorrect)
print("Correct ships:", correct)
print("Incorrect to correct ratio:", incorrect/correct)
print("Finished")
