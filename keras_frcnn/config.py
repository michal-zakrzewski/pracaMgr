from keras import backend as K


class Config:

    def __init__(self):

        self.verbose = True

        # model do wyciagania mapy cech
        self.network = 'resnet50'

        # domyslne ustawienia potrzeby obrotu/odbicia obrazka
        self.use_horizontal_flips = False
        self.use_vertical_flips = False
        self.rot_90 = False

        # domyslna wielkosc rozmiarow obszarow sieci RPN
        self.anchor_box_scales = [128, 256, 512]

        # domyslna wielkosc skal sieci RPN
        self.anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]

        # wartosc, do ktorej ma zostac mniejszona wielkosc
        self.im_size = 300

        # parametry okreslajace ekstrakcje wartosci z kanalow kolorow
        self.img_channel_mean = [103.939, 116.779, 123.68]
        self.img_scaling_factor = 1.0

        # liczba ROIs do jednego testu
        self.num_rois = 600

        # krok w RPN (moze zostac zmieniony przez konfiguracje samego testu)
        self.rpn_stride = 16

        self.balanced_classes = False

        # skalowania stdev
        self.std_scaling = 4.0
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

        # domyslne progi dla sieci RPN
        self.rpn_min_overlap = 0.3
        self.rpn_max_overlap = 0.7

        # progi dla nakladania sie ROIs
        self.classifier_min_overlap = 0.1
        self.classifier_max_overlap = 0.6

        # placeholder odwzorowania klas (generowany automatycznie)
        self.class_mapping = None

        # domyslna sciezka modelu
        self.model_path = 'model_frcnn.hdf5'
