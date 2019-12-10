import numpy as np
import math
from . import data_generators
import copy


def calc_iou(R, img_data, C, class_mapping):
    """
    Konwersja wspolrzednych z (x1,x2,y1,y2) na (x,y,w,h)
    :param R: wyjscie z warstwy non_max_suppression
    :param img_data: zdjecie
    :param C: konfiguracja przekazana w linii komend
    :param class_mapping: lista klas, ktore zostaly przekazane w ramach treningu
    :return: informacje o iou:
        X - wspolrzedne wejsciowe w formacie (x1,y1,w,h)
        Y1 - obszary odnoszace sie do liczby klas
        Y2 - wartosc probki treningowej dla ostatniej warstwy regresji (okresla, czy nalezy dodac do obliczen odpowiedni parametr)
    """

    bboxes = img_data['bboxes']
    (width, height) = (img_data['width'], img_data['height'])
    # wez wymiary obrazka dla zmian rozmiaru
    (resized_width, resized_height) = data_generators.get_new_img_size(width, height, C.im_size)

    gta = np.zeros((len(bboxes), 4))

    for bbox_num, bbox in enumerate(bboxes):
        # oblicz wspolrzedne GT i zmien rozmiar
        gta[bbox_num, 0] = int(round(bbox['x1'] * (resized_width / float(width))/C.rpn_stride))
        gta[bbox_num, 1] = int(round(bbox['x2'] * (resized_width / float(width))/C.rpn_stride))
        gta[bbox_num, 2] = int(round(bbox['y1'] * (resized_height / float(height))/C.rpn_stride))
        gta[bbox_num, 3] = int(round(bbox['y2'] * (resized_height / float(height))/C.rpn_stride))

    x_roi = []
    y_class_num = []
    y_class_regr_coords = []
    y_class_regr_label = []
    IoUs = [] # do celow statystycznych

    for ix in range(R.shape[0]):
        (x1, y1, x2, y2) = R[ix, :]
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        best_iou = 0.0
        best_bbox = -1
        for bbox_num in range(len(bboxes)):
            curr_iou = data_generators.iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1, y1, x2, y2])
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_bbox = bbox_num

        if best_iou < C.classifier_min_overlap:
                continue
        else:
            w = x2 - x1
            h = y2 - y1
            x_roi.append([x1, y1, w, h])
            IoUs.append(best_iou)

            if C.classifier_min_overlap <= best_iou < C.classifier_max_overlap:
                # hard negative mining
                cls_name = 'bg'
            elif C.classifier_max_overlap <= best_iou:
                cls_name = bboxes[best_bbox]['class']
                cxg = (gta[best_bbox, 0] + gta[best_bbox, 1]) / 2.0
                cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2.0

                cx = x1 + w / 2.0
                cy = y1 + h / 2.0

                tx = (cxg - cx) / float(w)
                ty = (cyg - cy) / float(h)
                tw = np.log((gta[best_bbox, 1] - gta[best_bbox, 0]) / float(w))
                th = np.log((gta[best_bbox, 3] - gta[best_bbox, 2]) / float(h))
            else:
                print('roi = {}'.format(best_iou))
                raise RuntimeError

        class_num = class_mapping[cls_name]
        class_label = len(class_mapping) * [0]
        class_label[class_num] = 1
        y_class_num.append(copy.deepcopy(class_label))
        coords = [0] * 4 * (len(class_mapping) - 1)
        labels = [0] * 4 * (len(class_mapping) - 1)
        if cls_name != 'bg':
            label_pos = 4 * class_num
            sx, sy, sw, sh = C.classifier_regr_std
            coords[label_pos:4+label_pos] = [sx*tx, sy*ty, sw*tw, sh*th]
            labels[label_pos:4+label_pos] = [1, 1, 1, 1]
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))
        else:
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))

    if len(x_roi) == 0:
        return None, None, None, None

    X = np.array(x_roi)
    Y1 = np.array(y_class_num)
    Y2 = np.concatenate([np.array(y_class_regr_label),np.array(y_class_regr_coords)],axis=1)

    return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0), IoUs


def apply_regr(x, y, w, h, tx, ty, tw, th):
    """
    Cel regresji to zblizenie wartosci T (poprzedni bbox) do X (obecny bbox).
    Po uzyskaniu wyniku prognozy rzeczywiste wspolrzedne obszaru predykcji mozna uzyskac za pomoca tej funkcji
    :param x: Wspolrzedna X srodka obszaru
    :param y: Wspolrzedna Y srodka obszaru
    :param w: Szerokosc obszaru
    :param h: Wysokosc obszaru
    :param tx: Wspolrzedna X poprzedniego srodka obszaru
    :param ty: Wspolrzedna Y poprzedniego srodka obszaru
    :param tw: Szerokosc poprzedniego obszaru
    :param th: Wysokosc poprzedniego obszaru
    :return: Nowe wspolrzedne (o ile dadza poprawe - w przypadku bledu zwraca poprzednie wartosci)
    """
    try:
        # wysrodkuj wspolrzedne
        cx = x + w/2.
        cy = y + h/2.
        # stale centrum do zmian wybiaru
        cx1 = tx * w + cx
        cy1 = ty * h + cy
        # stala wysokosc i szerokosc
        w1 = math.exp(tw) * w
        h1 = math.exp(th) * h
        # stala wartosc lewej gornej wspolrzednej
        x1 = cx1 - w1/2.
        y1 = cy1 - h1/2.
        # zaokraglanie
        x1 = int(round(x1))
        y1 = int(round(y1))
        w1 = int(round(w1))
        h1 = int(round(h1))

        return x1, y1, w1, h1

    except ValueError:
        return x, y, w, h
    except OverflowError:
        return x, y, w, h
    except Exception as e:
        print(e)
        return x, y, w, h


def apply_regr_np(X, T):
    """
    Funkcja dostosowywujaca pozycje GT za pomoca przewidywanej wartosci warstwy regresji RPN
    :param X: aktualne koordynaty
    :param T: parametry odpowiadajace obszarowi do poprawy
    :return: poprawione wspolrzedne obszaru
    """
    try:
        x = X[0, :, :]
        y = X[1, :, :]
        w = X[2, :, :]
        h = X[3, :, :]

        tx = T[0, :, :]
        ty = T[1, :, :]
        tw = T[2, :, :]
        th = T[3, :, :]

        cx = x + w/2.
        cy = y + h/2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy

        w1 = np.exp(tw.astype(np.float64)) * w
        h1 = np.exp(th.astype(np.float64)) * h
        x1 = cx1 - w1/2.
        y1 = cy1 - h1/2.

        x1 = np.round(x1)
        y1 = np.round(y1)
        w1 = np.round(w1)
        h1 = np.round(h1)
        return np.stack([x1, y1, w1, h1])
    except Exception as e:
        print(e)
        return X


def non_max_suppression_fast(boxes, probs, overlap_thresh=0.9, max_boxes=300):
    """
    Warstwa non_max_suprression
    :param boxes: obszar o strukturze (n,4) z odpowiadajacymi koordynatami
    :param probs: prawdopodobienstwo odpowiadajace kazdemu obszarowi
    :param max_boxes: maksymalna ilosc obszarow do sprawdzenia (uwaga, moze byc przekazana jako pamaetr z innej funkcji)
    :param overlap_thresh: prog okreslania pozytywnej probki (uwaga, moze byc przekazana jako pamaetr z innej funkcji)
    :return: wyjscie z warstwy non_max_suppression
    """
    # jesli nie ma zadnych bboxow, zwrot pusta liste i wyjdz
    if len(boxes) == 0:
        return []

    # wyciagnij wspolrzedne bboxa
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]      

    # zmien rodzaj zmiennej z int na float (dalej sa dzielenia)
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # definicja listy uzytych indeksow
    pick = []

    # oblicz wielkosc obszaru
    area = (x2 - x1) * (y2 - y1)

    # posortuj bboxy
    idxs = np.argsort(probs)

    # petla sprawdzajaca wszystkie indeksy
    while len(idxs) > 0:
        # przenies ostatni indeks z listy indeksow wejsciowych i wstaw
        # do listy indeksow sprawdzonych
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # oblicz iloczyn obszarow
        xx1_int = np.maximum(x1[i], x1[idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int

        # oblicz sume obszarow
        area_union = area[i] + area[idxs[:last]] - area_int

        # oblicz wsp nakladania sie obszarow
        overlap = area_int/(area_union + 1e-3)

        # usun wszystkie zuzyte/nie nakladajace sie indeksy (zeby nie byly brane ponownie pod uwage)
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break

    # zwroc tylko te bboxy, ktore zostaly uzyte zamieniajac ich pozycje na inty
    boxes = boxes[pick].astype("int")
    probs = probs[pick]
    return boxes, probs


def rpn_to_roi(rpn_layer, regr_layer, C, dim_ordering, use_regr=True, max_boxes=300, overlap_thresh=0.8):
    """
    Funkcja zmieniajaca wygenerowane obszary z sieci RPN jako ROI (jako wyjscie warstwy non_max_suppression) do celow testowych skutecznosci treningu
    :param rpn_layer: wyjscie modelu predykcyjnego sieci RPN - warstwa klasyfikacji
    :param regr_layer: wyjscie modelu predyjcyjnego sieci RPN - warstwa regresji
    :param C: konfiguracja przekazana z linii komand
    :param dim_ordering: parametr okreslajacy ulozenie parametrow (czy najpierw ida wymiary i na koncu kanaly jak w theano, czy kanaly pierwsze, jak w tf)
    :param use_regr: parametr okreslajacy czy wspolrzedne moga byc zmieniane w zaleznosci od znalezionych gradientow
    :param max_boxes: maksymalna ilosc obszarow do sprawdzenia (uwaga, moze byc przekazana jako pamaetr z innej funkcji)
    :param overlap_thresh: prog okreslania pozytywnej probki (uwaga, moze byc przekazana jako pamaetr z innej funkcji)
    :return: ROI dla obszarow z sieci RPN
    """

    regr_layer = regr_layer / C.std_scaling

    anchor_sizes = C.anchor_box_scales
    anchor_ratios = C.anchor_box_ratios

    assert rpn_layer.shape[0] == 1

    if dim_ordering == 'th':
        (rows,cols) = rpn_layer.shape[2:]

    elif dim_ordering == 'tf':
        (rows, cols) = rpn_layer.shape[1:3]

    curr_layer = 0
    if dim_ordering == 'tf':
        A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3]))
    elif dim_ordering == 'th':
        A = np.zeros((4, rpn_layer.shape[2], rpn_layer.shape[3], rpn_layer.shape[1]))

    # wymiary [128, 256, 512]
    for anchor_size in anchor_sizes:
        # ratio, 1:2, 2:1, 1:1
        for anchor_ratio in anchor_ratios:

            # znajdz szerokosc i wysokosc GT na mapie cech
            anchor_x = (anchor_size * anchor_ratio[0])/C.rpn_stride
            anchor_y = (anchor_size * anchor_ratio[1])/C.rpn_stride
            if dim_ordering == 'th':
                regr = regr_layer[0, 4 * curr_layer:4 * curr_layer + 4, :, :]
            else:
                regr = regr_layer[0, :, :, 4 * curr_layer:4 * curr_layer + 4]
                regr = np.transpose(regr, (2, 0, 1))

            X, Y = np.meshgrid(np.arange(cols),np. arange(rows))

            # zapisz wspolrzedne GT (x,y,w,h)
            A[0, :, :, curr_layer] = X - anchor_x/2
            A[1, :, :, curr_layer] = Y - anchor_y/2
            A[2, :, :, curr_layer] = anchor_x
            A[3, :, :, curr_layer] = anchor_y

            if use_regr:
                A[:, :, :, curr_layer] = apply_regr_np(A[:, :, :, curr_layer], regr)

            # usun koordynaty niepasujace
            # np.maximum(1, []) ustawia wartosc 1 w tabeli [] dla wartosci mniejszych niz 1
            A[2, :, :, curr_layer] = np.maximum(1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.maximum(1, A[3, :, :, curr_layer])
            A[2, :, :, curr_layer] += A[0, :, :, curr_layer]
            A[3, :, :, curr_layer] += A[1, :, :, curr_layer]

            # sprawdzenie, czy wartosci nie wychodza za obrazek
            A[0, :, :, curr_layer] = np.maximum(0, A[0, :, :, curr_layer])
            A[1, :, :, curr_layer] = np.maximum(0, A[1, :, :, curr_layer])
            A[2, :, :, curr_layer] = np.minimum(cols-1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.minimum(rows-1, A[3, :, :, curr_layer])

            # nastepna warstwa
            curr_layer += 1
    # obszar o strukturze (n,4) z odpowiadajacymi koordynatami oraz ich prawdopodobienstwo
    all_boxes = np.reshape(A.transpose((0, 3, 1,2)), (4, -1)).transpose((1, 0))
    all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1))

    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]

    idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))

    all_boxes = np.delete(all_boxes, idxs, 0)
    all_probs = np.delete(all_probs, idxs, 0)

    result = non_max_suppression_fast(all_boxes, all_probs, overlap_thresh=overlap_thresh, max_boxes=max_boxes)[0]

    return result
