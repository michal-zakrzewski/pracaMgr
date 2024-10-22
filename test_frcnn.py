# -*- coding: utf-8 -*-
from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
import keras_frcnn.resnet as nn
from sys import platform
import shutil
import datetime

if platform == "linux" or platform == "linux2":
    from IPython.core.display import display
    path = str("/content/pracaMgr/input")
elif platform == "darwin":
    path = str("./input")
elif platform == "win32":
    path = str("../input")

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("-n", "--num_rois", dest="num_rois",
                  help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_filename", dest="config_filename",
                  help="Location to read the metadata related to the training (generated when training).",
                  default="config.pickle")
parser.add_option("--input_weight_path", dest="input_weight_path", help="Input path for weights.",
                  default="./model_frcnn.hdf5")

(options, args) = parser.parse_args()

if not options.test_path:  # if filename is not given
    parser.error('Error: path to test data must be specified. Pass --path to command line')

config_output_filename = options.config_filename

# wczytanie pliku konfiguracyjnego wag
with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)

# stworzenie plikow do zapisywania wynikow testu
with open(path + "/submission.csv", "w") as f:
    f.write('ImageId,EncodedPixels\n')
with open(path + "/moreShips.csv", "w") as f:
    f.write('ImageId,numberOfShips\n')
with open(path + "/possibleCollisions.csv", "w") as f:
    f.write('ImageId,numberOfShips\n')

# wylacz zmiane plikow podczas testu
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

img_path = options.test_path


def format_img_size(img, C):
    """
    Funkcja formatujaca rozmiar obrazka bazujac na konfiguracji
    :param img: zdjecie
    :param C: konfiguracja
    :return: zdjecie wraz ze skala, ktora trzeba uzyc do zmiany zdjecia
    """
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_channels(img, C):
    """
    Funkcja formatujaca kanaly na zdjeciu bazujac na konfiguracji
    :param img: zdjecie
    :param C: konfiguracja
    :return: zdjecie z rozpisaniem informacji o kanalach
    """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """
    Funckja formatujaca obraz dla modelu predykcyjnego bazujacego na konfiguracji
    :param img: zdjecie
    :param C: konfiguracja
    :return: zdjecie (z wyciagnietymi informacjami o kanalach) oraz skala uzyta w konfiguracji
    """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio


def get_real_coordinates(ratio, x1, y1, x2, y2):
    """
    Metoda zmieniajaca wspolrzedne BBoxa do oryginalnego rozmiaru
    :param ratio: skala uzyta podczas analizy zdjecia
    :param x1: Wsp X lewego gornego rogu
    :param y1: Wsp X prawego dolnego rogu
    :param x2: Wsp Y lewego gornego rogu
    :param y2: Wsp Y prawego dolnego rogu
    :return: Rzeczywiste polozenie wsp BBoxa na oryginalnym obrazie
    """
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return real_x1, real_y1, real_x2, real_y2


# Following function returns true if the rectangles are overlaping
def overlap_checker(x1, y1, x2, y2, all_coord):
    """
    Funkcja sprawdzajaca, czy obszary nie nakrywaja sie ze soba (generuje to bledy podczas
        sprawdzania pliku na portalu Kaggle
    :param x1: Wsp X lewego gornego rogu
    :param y1: Wsp X prawego dolnego rogu
    :param x2: Wsp Y lewego gornego rogu
    :param y2: Wsp Y prawego dolnego rogu
    :param all_coord: Lista wszystkich wsp poprzednich statkow dla danego zdjecia
    :return: Prawda/Falsz o nakladaniu sie zdjec
    """
    overlaps = False
    i = 0
    start = 0
    for i in range(int(len(all_coord)/4)):
        b = all_coord[start:start + 4]
        start += 4
        try:
            if (max(b[0], b[2]) <= min(x1, x2) or max(x1, x2) <= min(b[0], b[2]) or max(b[1], b[3]) <= min(y1, y2) or max(y1, y2) <= min(b[1], b[3])):
                if not (min(x1, x2) <= min(b[0], b[2]) and min(y1, y2) <= min(b[1], b[3]) and max(x1, x2) >= max(b[0], b[2]) and max(y1, y2) >= max(b[1], b[3])):
                    if not (min(b[0], b[2]) <= min(x1, x2) and min(b[1], b[3]) <= min(y1, y2) and max(b[0], b[2]) >= max(x1, x2) and max(b[1], b[3]) >= max(y1, y2)):
                        overlaps = False
                    else:
                        return True
                else:
                    return True
            else:
                return True
        except TypeError:
            overlaps = False
    if not overlaps:
        return False


class_mapping = C.class_mapping

if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = int(options.num_rois)

num_features = 1024

if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
    input_shape_features = (num_features, None, None)
else:
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# definicja ekstraktora mapy cech (Resnet)
shared_layers = nn.nn_base(img_input, trainable=True)

# definicja sieci RPN bazujaca na Resnet
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

# definicja sieci klasyfikujacej
classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

# wczytanie pliku wag
model_path = options.input_weight_path
print('Loading weights from {}'.format(model_path))
model_rpn.load_weights(model_path, by_name=True)
model_classifier.load_weights(model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

all_imgs = []

classes = {}

# ustawienie progu pozytywnej probki
bbox_threshold = 0.8

visualise = True
counter = 0
foundCounter = 0
if platform == "linux" or platform == "linux2":
    if not os.path.exists('/content/drive/My Drive/pracaMgr/results_imgs'):
        os.mkdir('/content/drive/My Drive/pracaMgr/results_imgs')
    if not os.path.exists('/content/drive/My Drive/pracaMgr/moreShipsImages'):
        os.mkdir('/content/drive/My Drive/pracaMgr/moreShipsImages')

for idx, img_name in enumerate(sorted(os.listdir(img_path))):
    if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
        continue
    all_coordinates = tuple()
    saveValue = False
    print(img_name)
    st = time.time()
    filepath = os.path.join(img_path, img_name)

    img = cv2.imread(filepath)

    X, ratio = format_img(img, C)

    if K.image_dim_ordering() == 'tf':
        X = np.transpose(X, (0, 2, 3, 1))

    # wydobycie mapy cech obrazu i wyjscia z sieci RPN
    # Y1 - prawdopodobienstwo zawierania kotwicy
    # Y2 - gradient regresji odpowiadajacy kotwicy
    # F - mapa obiektow po konwolucji
    [Y1, Y2, F] = model_rpn.predict(X)

    # przeksztalc rpn na obszary (bbox)
    R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.5)

    # zamiana wsp (x1,y1,x2,y2) na (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    # apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}

    # iteruj po wszystkich znalezionych obszarach
    for jk in range(R.shape[0] // C.num_rois + 1):
        # zapisz wszystkie num_rois ROIs
        ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
        # Jesli nie ma zadnych obszarow, przerwij
        if ROIs.shape[1] == 0:
            break

        # Jesli kolejnego nie ma, wypelnij rois samymi 0
        if jk == R.shape[0] // C.num_rois:
            # pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        # Wyslij dane do sieci klasyfikacyjnej, oszacuj granice obszarow
        # P_cls - prawdopodobienstwo, ze granica nalezy do okreslonej kategorii
        # P_regr - gradient regresji granicy odpowiadajacy kazdej kategorii
        # F - konwolucyjna mapa obiektow uzyskana przez siec RPN
        # ROIs - wstepnie wybrany obszar analizy
        [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

        for ii in range(P_cls.shape[1]):

            # Jesli obszar jest mniejszy, niz rozmiar w konfiguracji, bbox jest niepoprawny => jest tlem
            # pomin ten obszar
            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            # sprawdz do ktorej klasy jest max prawdopodobienstwo przynaleznosci
            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

            # zapis bboxy i prawodpodobienstwa
            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            # oblicz obecne wsp propozycji obszaru
            (x, y, w, h) = ROIs[0, ii, :]

            # wyznacz ktora pozycja ma max prawdopodobienstwo
            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                # zdobadz poprzednie polozenie propozycji obszaru
                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                # przypomnienie: a /= b oznacza a = a/b
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                # oblicz nowe pozycje wsp
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            # zapisz rzeczywiste wsp obecnego obszaru
            bboxes[cls_name].append(
                [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
            # oraz odpowiadajace jej prawodpodobienstwo
            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    all_dets = []
    firstPixel = 1
    thick = 0
    lastPixel = 0

    # dla wszystkich klas obecnych bboxow
    for key in bboxes:
        # wsp bboxow
        bbox = np.array(bboxes[key])

        # zlacz ze soba obszary bedace obok siebie korzystajac z alg non max suppression
        new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.8)
        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk, :]

            # oblicz rzeczywiste polozenie wsp na oryginalnym obrazie
            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
            # aby uniknac nakladania sie obszarow zmniejsz wsp o 1
            if abs(real_x2-real_x1) >= 3 and abs(real_y2 - real_y1) >= 3:
                if real_x1 < real_x2:
                    real_x1 += 1
                    real_x2 -= 1
                else:
                    real_x2 += 1
                    real_x1 -= 1
                if real_y1 < real_y2:
                    real_y1 += 1
                    real_y2 -= 1
                else:
                    real_y1 -= 1
                    real_y2 += 1
            # upewnij sie, ze absolutnie zadna pozycja nie wypadla poza obraz
            if real_x1 > 767:
                real_x1 = 767
            if real_x2 > 767:
                real_x2 = 767
            if real_y1 > 767:
                real_y1 = 767
            if real_y2 > 767:
                real_y2 = 767
            if real_x1 < 0:
                real_x1 = 0
            if real_x2 < 0:
                real_x2 = 0
            if real_y1 < 0:
                real_y1 = 0
            if real_y2 < 0:
                real_y2 = 0
            # upewnij sie, ze dany obszar nie pokrywa sie z istniejacym juz wczesniej
            if all_coordinates:
                if overlap_checker(real_x1, real_y1, real_x2, real_y2, all_coordinates):
                    saveValue = True
                    # jesli tak - pomin obecny
                    continue
            # kodowanie pozycji bboxa na RLE
            all_coordinates = all_coordinates + tuple([real_x1, real_y1, real_x2, real_y2])
            encodedPixels = ''
            i = 1
            firstPixel = real_x1 * 768 + real_y1
            thick = real_y2 - real_y1
            lastPixel = real_x2 * 768 + real_y2
            if firstPixel == 0:
                firstPixel = 1
            # zapisanie pozycji kolejnych pikseli w formacie RLE jako string
            encodedPixels += str(firstPixel)
            encodedPixels += " "
            encodedPixels += str(thick)
            encodedPixels += " "
            # parametr i to liczba linii
            while True:
                nextPixel = firstPixel + 768 * i
                checkLastPixel = nextPixel + thick
                if checkLastPixel >= lastPixel:
                    break
                i += 1
                encodedPixels += str(nextPixel)
                encodedPixels += " "
                encodedPixels += str(thick)
                encodedPixels += " "
            # zapisz wynik do pliku submission.csv
            with open(path + "/submission.csv", "a") as f:
                print(img_name, encodedPixels, sep=',', file=f)

            # fragment kodu obdpowiadajacy za narysowanie prostokata wokol statku i dodanie tekstu
            # oznaczajacego klase wraz z prawdopodobienstwem przyporzadkowania do tej klasy
            cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2),
                          (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])), 2)

            textLabel = 'ship: {}%'.format(int(100 * new_probs[jk]))
            all_dets.append((key, 100 * new_probs[jk]))

            (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
            textOrg = (real_x1, real_y1)

            cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                         (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 2)
            cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                         (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
            cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

    # zapisz statystyki zwiazane z analiza obrazu
    print('Elapsed time = {}'.format(time.time() - st))
    print(all_dets)
    if len(all_dets) > 0:
        if platform == "linux" or platform == "linux2":
            try:
                cv2.imwrite('/content/drive/My Drive/pracaMgr/results_imgs/{}'.format(img_name), img)
            except Exception as e:
                print("No possibility to save an image!")
                print(e)
        else:
            if not os.path.exists(path + '/results_imgs'):
                os.mkdir(path + '/results_imgs')
            try:
                cv2.imwrite(path + 'results_imgs/{}.png'.format(img_name), img)
            except Exception as e:
                print("No possibility to save an image!")
                print(e)

    else:
        with open(path + "/submission.csv", "a") as f:
            print(img_name, "", sep=',', file=f)
    if len(all_dets) > 1:
        with open(path + "/moreShips.csv", "a") as f:
            print(img_name, len(all_dets), sep=',', file=f)
        if platform == "linux" or platform == "linux2":
            try:
                # zapisz obraz z wieksza iloscia statkow na zewnetrznym folderze
                cv2.imwrite('/content/drive/My Drive/pracaMgr/moreShipsImages/{}'.format(img_name), img)
            except Exception as e:
                print("No possibility to save an image!")
                print(e)

    counter += 1
    foundCounter += 1

    # debug only - do sprawdzenia potencjalnych kolizji
    if saveValue and len(all_dets) > 1:
        with open(path + "/possibleCollisions.csv", "a") as f:
            print(img_name, len(all_dets), sep=',', file=f)

    # zapisanie plikow na dysku Google
    if platform == "linux" or platform == "linux2" and counter == 20:
        shutil.copy(path + "/submission.csv",
                    "/content/drive/My Drive/pracaMgr/submission.csv")
        shutil.copy(path + "/moreShips.csv",
                    "/content/drive/My Drive/pracaMgr/moreShips.csv")
        shutil.copy(path + "/possibleCollisions.csv",
                    "/content/drive/My Drive/pracaMgr/possibleCollisions.csv")
        counter = 0

# zapisanie ostatecznych wynikow na dysku Google
if platform == "linux" or platform == "linux2":
    shutil.copy(path + "/submission.csv",
                "/content/drive/My Drive/pracaMgr/submission" + str(datetime.date.today()) + "final.csv")
    shutil.copy(path + "/moreShips.csv",
                "/content/drive/My Drive/pracaMgr/moreShips" + str(datetime.date.today()) + "final.csv")
    shutil.copy(path + "/possibleCollisions.csv",
                "/content/drive/My Drive/pracaMgr/possibleCollisions" + str(datetime.date.today()) + "final.csv")
print("Finished")
print("Found " + str(foundCounter) + " ships")
