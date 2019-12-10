from __future__ import absolute_import
import numpy as np
import cv2
import random
import copy
from . import data_augment
import threading
import itertools


def union(au, bu, area_intersection):
    """
    Funkcja obliczajaca sume obszarow
    :param au: obszar a, wsp. (x1,y1,x2,y2)
    :param bu: obszar b, wsp. (x1,y1,x2,y2)
    :param area_intersection: pole sumy
    :return:
    """
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai, bi):
    """
    Funkcja obliczajaca iloczyn obszarow
    :param ai: obszar a, wsp. (x1,y1,x2,y2)
    :param bi: obszar b, wsp. (x1,y1,x2,y2)
    :return: pole iloczynu
    """
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w*h


def iou(a, b):
    """
    Obliczanie wspolczynnika Jaccarda
    Współczynnik Jaccarda mierzy podobieństwo między dwoma zbiorami
    :param a: obszar 1, wsp. (x1,y1,x2,y2)
    :param b: obszar 2, wsp. (x1,y1,x2,y2)
    :return: % nakladania sie obszarow a i b
    """
    # a and b should be (x1,y1,x2,y2)

    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)


def get_new_img_size(width, height, img_min_side=600):
    """
    Funckaj odpowiadajaca za zmiane rozmiaru zdjecia
    :param width: zadana szerokosc zdjecia
    :param height: zadana wysokosc zdjecia
    :param img_min_side: zadana wielkosc boku
    :return: zmieniona szerokosc, zmieniona wysokosc
    """
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_width, resized_height


# for balanced class
class SampleSelector:
    def __init__(self, class_count):
        """
        Konstruktor klasy (uruchamiany zawsze, gdy jest tworzony nowy obiekt tej klasy)
        :param class_count: liczba klas
        """
        # ignore classes that have zero samples
        self.classes = [b for b in class_count.keys() if class_count[b] > 0]
        self.class_cycle = itertools.cycle(self.classes)
        self.curr_class = next(self.class_cycle)

    def skip_sample_for_balanced_class(self, img_data):
        """
        Funkcja balansujaca ilosc klas treningu (gdy jest wieksza ilosc, niz 2)
        :param img_data: dane wejsciowe
        :return: prawda/falsz, czy jest klasa na zdjeciu
        """

        class_in_img = False

        for bbox in img_data['bboxes']:

            cls_name = bbox['class']

            if cls_name == self.curr_class:
                class_in_img = True
                self.curr_class = next(self.class_cycle)
                break

        if class_in_img:
            return False
        else:
            return True


def calc_rpn(C, img_data, width, height, resized_width, resized_height, img_length_calc_function):
    """
    Funkcja obliczajaca i analizujaca wszystkie GT dla zdjecia do obliczenia IOU
    :param C: konfiguracja
    :param img_data: dane wejsciowe
    :param width: oryginalna szerokosc zdjecia
    :param height: oryginalna wysokosc zdjecia
    :param resized_width: zmieniona szerokosc zdjecia
    :param resized_height: zmieniona wysokosc zdjecia
    :param img_length_calc_function: funkcja bibiliteki Keras
    :return: np.copy(y_rpn_cls): czy obraz zaiwera szukana klase
    :return: np.copy(y_rpn_regr): odpowiadajaca gradient dla tego obrazu
    """

    downscale = float(C.rpn_stride)
    anchor_sizes = C.anchor_box_scales
    anchor_ratios = C.anchor_box_ratios
    num_anchors = len(anchor_sizes) * len(anchor_ratios)

    # oblicz wielkosc wyjsciowa cechy
    (output_width, output_height) = img_length_calc_function(resized_width, resized_height)

    n_anchratios = len(anchor_ratios)

    # initialise empty output objectives
    y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
    y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
    y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

    num_bboxes = len(img_data['bboxes'])

    num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
    best_anchor_for_bbox = -1 * np.ones((num_bboxes, 4)).astype(int)
    best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
    best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
    best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

    # oblicz wsp GT i zmien rozmiar obrazu zgodnie z potrzebami
    gta = np.zeros((num_bboxes, 4))
    for bbox_num, bbox in enumerate(img_data['bboxes']):
        # oblicz wsp tego GT i zmien rozmiar obrazu zgodnie z potrzebami
        gta[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))
        gta[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
        gta[bbox_num, 2] = bbox['y1'] * (resized_height / float(height))
        gta[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))

    # przejrzyj wszystkie mozliwosci grupy rozmiarow
    # rozmiar boku: (128,256,512)
    # skale: (1:1,2:1,1:2)
    for anchor_size_idx in range(len(anchor_sizes)):
        for anchor_ratio_idx in range(n_anchratios):
            anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
            anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]

            for ix in range(output_width):
                # koordynaty x-owe aktualnego GT bboxa
                x1_anc = downscale * (ix + 0.5) - anchor_x / 2
                x2_anc = downscale * (ix + 0.5) + anchor_x / 2

                # zignoruj obszar, ktory wychodzi poza obraz
                if x1_anc < 0 or x2_anc > resized_width:
                    continue

                for jy in range(output_height):

                    # koordynaty y-kowe ktualnego GT bboxa
                    y1_anc = downscale * (jy + 0.5) - anchor_y / 2
                    y2_anc = downscale * (jy + 0.5) + anchor_y / 2

                    # zignoruj obszar, ktory wychodzi poza obraz
                    if y1_anc < 0 or y2_anc > resized_height:
                        continue

                    # ten parametr wskazuje rodzaj kotwica (obszar zadany przez uzytkownika) powinien byc uzyty
                    # domyslnie negatywny
                    bbox_type = 'neg'

                    # parametr okreslajacy IOU pomiedzy GT i obecnym obszarem
                    best_iou_for_loc = 0.0

                    # cel treningu - doprowadzenie sygnalow wyjsciowych do sieci do wejscia tak blisko, jak to mozliwe
                    for bbox_num in range(num_bboxes):

                        # oblicz IOU dla danego GT i obecnie badanego obszaru
                        curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1_anc, y1_anc, x2_anc, y2_anc])
                        # oblicz funkcje strat regresji obszaru (cel)
                        # ponizsze parametry do korygowania polozenia anchora
                        # tx - stosunek polozenia x srodka dwoch pol do szerokosci pola zdefiniowanego przez usera
                        # ty - jw. dla zmiennej y
                        # tw - stosunek szerokosci pola badanego do pola zdefiniowanego przez uzytkownika
                        # th - jw dla wysokosc pola
                        if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap:
                            cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
                            cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
                            cxa = (x1_anc + x2_anc)/2.0
                            cya = (y1_anc + y2_anc)/2.0

                            tx = (cx - cxa) / (x2_anc - x1_anc)
                            ty = (cy - cya) / (y2_anc - y1_anc)
                            tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
                            th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))

                        if img_data['bboxes'][bbox_num]['class'] != 'bg':

                            # wszystkie GT powinny zostac zmapowane do obszaru uzytkownika aby okreslic ktory najlepszy
                            if curr_iou > best_iou_for_bbox[bbox_num]:
                                best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
                                best_iou_for_bbox[bbox_num] = curr_iou
                                best_x_for_bbox[bbox_num,:] = [x1_anc, x2_anc, y1_anc, y2_anc]
                                best_dx_for_bbox[bbox_num,:] = [tx, ty, tw, th]

                            # okres przyklad jako pozytywny, jesli IOU>0.7 (niezaleznie, czy byl lepszy przyklad)
                            if curr_iou > C.rpn_max_overlap:
                                bbox_type = 'pos'
                                num_anchors_for_bbox[bbox_num] += 1
                                # aktualizacja wartosci funkcji strat, jesli ten IOU jest najlepszy dla danego obszaru
                                if curr_iou > best_iou_for_loc:
                                    best_iou_for_loc = curr_iou
                                    best_regr = (tx, ty, tw, th)

                            # jesli IOU E (0.3,0.7) - nie uzywaj do treningu
                            if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
                                # oznacz jako neutralny
                                if bbox_type != 'pos':
                                    bbox_type = 'neutral'

                    # w zaleznosci od IOU wrzuc odpowiednie bboxy do odpowiednich parametrow
                    if bbox_type == 'neg':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif bbox_type == 'neutral':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif bbox_type == 'pos':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
                        y_rpn_regr[jy, ix, start:start+4] = best_regr

    # sprawdz, czy kazdy bbox ma przynajmniej jeden pozytywny region po RPN

    for idx in range(num_anchors_for_bbox.shape[0]):
        if num_anchors_for_bbox[idx] == 0:
            # zaden bbox z IOU>0 - przerwij
            if best_anchor_for_bbox[idx, 0] == -1:
                continue
            y_is_box_valid[
                best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
                best_anchor_for_bbox[idx,3]] = 1
            y_rpn_overlap[
                best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
                best_anchor_for_bbox[idx,3]] = 1
            start = 4 * (best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3])
            y_rpn_regr[
                best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], start:start+4] = best_dx_for_bbox[idx, :]

    y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

    y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

    y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

    pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
    neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

    num_pos = len(pos_locs[0])

    # jesli jest znacznie za duzo negatywnych przykladow, ogranicz ich liczbe do 256 (balansowanie)
    num_regions = 256

    if len(pos_locs[0]) > num_regions/2:
        val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions/2)
        y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
        num_pos = num_regions/2

    if len(neg_locs[0]) + num_pos > num_regions:
        val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
        y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

    return np.copy(y_rpn_cls), np.copy(y_rpn_regr)


class threadsafe_iter:
    """
    Klasa bierze iterator/generator i ubezpiecznia go w przypadku dzialania na wielu watkach
    poprzez uzycie wywolania metody next
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """
    Dekorator przyjmujacy funkcje generatora do bezpieczenstwa watkow.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


def get_anchor_gt(all_img_data, class_count, C, img_length_calc_function, backend, mode='train'):
    """
    Funkcja okreslajaca polozenie bboxa anchora
    :param all_img_data: sparsowane dane wejsciowe
    :param class_count: liczba klas
    :param C: konfiguracja
    :param img_length_calc_function: parametr architektury Resnet
    :param backend: parametr dedykowany Tensorflow
    :param mode: Parametr przyjmujacy wartosci train/val (w zaleznosi od trybu)
    :return: macierz opisujaca zdjecie wejsciowe, obliczone wartosci funkcji strat oraz tryb
    """
    sample_selector = SampleSelector(class_count)

    while True:
        if mode == 'train':
            random.shuffle(all_img_data)

        for img_data in all_img_data:
            try:

                if C.balanced_classes and sample_selector.skip_sample_for_balanced_class(img_data):
                    continue

                # read in image, and optionally add augmentation

                if mode == 'train':
                    img_data_aug, x_img = data_augment.augment(img_data, C, augment=True)
                else:
                    img_data_aug, x_img = data_augment.augment(img_data, C, augment=False)

                (width, height) = (img_data_aug['width'], img_data_aug['height'])
                (rows, cols, _) = x_img.shape

                assert cols == width
                assert rows == height

                # get image dimensions for resizing
                (resized_width, resized_height) = get_new_img_size(width, height, C.im_size)

                # resize the image so that smalles side is length = 600px
                x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

                try:
                    # rpn ground-truth cls, reg
                    y_rpn_cls, y_rpn_regr = calc_rpn(C, img_data_aug, width, height, resized_width, resized_height, img_length_calc_function)
                except:
                    continue

                # Zero-center by mean pixel, and preprocess image

                x_img = x_img[:, :, (2, 1, 0)]  # BGR -> RGB
                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]
                x_img /= C.img_scaling_factor

                x_img = np.transpose(x_img, (2, 0, 1))
                x_img = np.expand_dims(x_img, axis=0)

                y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= C.std_scaling

                if backend == 'tf':
                    x_img = np.transpose(x_img, (0, 2, 3, 1))
                    y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
                    y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

                yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug

            except Exception as e:
                print(e)
                continue
