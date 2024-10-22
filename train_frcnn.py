# -*- coding: utf-8 -*-
from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import os
import shutil

import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils
from keras.callbacks import TensorBoard
from keras_frcnn.simple_parser import get_data
from sys import platform
from keras_frcnn import resnet as nn

if platform == "linux" or platform == "linux2":
    from IPython.core.display import display

    path = str("/content/pracaMgr/input")
elif platform == "darwin":
    path = str("./input")
elif platform == "win32":
    path = str("../input")


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


# Wczytanie danych wprowadzonych jako parametry w konsoli
sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="train_path",
                  help="Path to training data.")
parser.add_option("--num_epochs", dest="num_epochs",
                  help="Number of epochs.", default=40)
parser.add_option("--epoch_length", dest="epoch_length",
                  help="Number of length", default=1000)
parser.add_option("--num_rois", dest="num_rois",
                  help="Number of RoIs", default=32)
parser.add_option("--config_filename", dest="config_filename",
                  help="Location to store all the metadata related to the training (to be used when testing).",
                  default="config.pickle")
parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=false).",
                  action="store_true", default=False)
parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).",
                  action="store_true", default=False)
parser.add_option("--rot", "--rot_90", dest="rot_90",
                  help="Augment with 90 degree rotations in training. (Default=false).",
                  action="store_true", default=False)
parser.add_option("--output_weight_path", dest="output_weight_path",
                  help="Output path for weights.", default=path + '/weights.hdf5')
parser.add_option("--input_weight_path",
                  dest="input_weight_path", help="Input path for weights.")

(options, args) = parser.parse_args()

if not options.train_path:  # if filename is not given
    parser.error(
        'Error: path to training data must be specified. Pass --path to command line')

# przekazanie ustawien z linii polecen i uzycie ich w konfiguracji
C = config.Config()

C.model_path = options.output_weight_path
C.num_rois = int(options.num_rois)

# wylaczanie modyfikacji obrazu podczas testu
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

# sprawdzenie, czy w linii polecen przekazano sciezke do pliku wag
if options.input_weight_path:
    C.base_net_weights = options.input_weight_path
else:
    # ustawienie sciezki do wag
    C.base_net_weights = nn.get_weight_path()

# parser
all_imgs, classes_count, class_mapping = get_data(options.train_path)

# tlo (bg)
if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))

config_output_filename = options.config_filename

with open(config_output_filename, 'wb') as config_f:
    pickle.dump(C, config_f)
    print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
        config_output_filename))

random.shuffle(all_imgs)

num_imgs = len(all_imgs)

# Wczytanie flag oznaczanych zbiór jako treningowy/walidacyjny/testowy (sprawdzający accuracy podczas treningu)
train_imgs = [s for s in all_imgs if s['imageset'] == 'train']
val_imgs = [s for s in all_imgs if s['imageset'] == 'val']
test_imgs = [s for s in all_imgs if s['imageset'] == 'test']

print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))
print('Num test samples {}'.format(len(test_imgs)))

if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
else:
    input_shape_img = (None, None, 3)

# input placeholder
img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))

# base network (definicja architektury do wyciągania mapy cech z obrazu)
shared_layers = nn.nn_base(img_input, trainable=True)

# definicja RPN
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)

# sięc do klasyfikacji
classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(
    classes_count), trainable=True)

model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

# definicja modelu, ktory przetrzymuje obie sieci, RPN i klasyfikator, do zapisywania/wczytania wag modelu
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

try:
    # wczytanie modelu (jesli zostal przekazany)
    print('loading weights from {}'.format(C.base_net_weights))
    model_rpn.load_weights(C.base_net_weights, by_name=True)
    model_classifier.load_weights(C.base_net_weights, by_name=True)
except:
    print('Could not load pretrained model weights. Weights can be found in the keras application folder \
        https://github.com/fchollet/keras/tree/master/keras/applications')

# parametry zwiazane z nauka
optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(
    num_anchors), losses.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(
    len(classes_count) - 1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')

# Tensorboard log
log_path = './logs'
if not os.path.isdir(log_path):
    os.mkdir(log_path)

# Tensorboard log
callback = TensorBoard(log_path)
callback.set_model(model_all)

epoch_length = int(options.epoch_length)
num_epochs = int(options.num_epochs)
iter_num = 0
train_step = 0
export_counter = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf

class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('Starting training')

# Do logowania na Google Drive wynikow
if os.path.exists("/content/drive/My Drive/pracaMgr/Weights/config.pickle"):
    os.remove("/content/drive/My Drive/pracaMgr/Weights/config.pickle")
if os.path.exists("/content/drive/My Drive/pracaMgr/losses_values.csv"):
    os.remove("/content/drive/My Drive/pracaMgr/losses_values.csv")
if os.path.exists(path + "/losses_values.csv"):
    os.remove(path + "/losses_values.csv")
with open(path + "/losses_values.csv", "w") as f:
    f.write('epoch_num,curr_loss,loss_rpn_regr,rpn_loss,time\n')
with open(path + "/rpn_loss.csv", "w") as f:
    f.write('train_step,rpn_cls,rpn_regr,detector_cls,detector_regr,total\n')

# Petla z treningiem
for epoch_num in range(num_epochs):

    progbar = generic_utils.Progbar(epoch_length)  # keras progress bar
    print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

    # okreslenie polozenia GT
    data_gen_train = data_generators.get_anchor_gt(
        train_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')
    data_gen_val = data_generators.get_anchor_gt(
        val_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='val')
    data_gen_test = data_generators.get_anchor_gt(
        test_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='test')

    while True:
        # Ten fragment wyrzuca blad, jesli RPN nie jest w stanie wyrzucic zadnego regionu dla pozytywnej probki
        if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
            mean_overlapping_bboxes = float(
                # pozytywne obszary/wszystkie obszary
                sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
            rpn_accuracy_rpn_monitor = []
            print('\nAverage number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
                mean_overlapping_bboxes, epoch_length))
            if mean_overlapping_bboxes == 0:
                print(
                    'RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

        # uzyskaj informacje o zdjeciu dla RPN: X wejscie, Y wyjscie
        X, Y, img_data = next(data_gen_train)

        # trenuj model - X to wejscie (obraz), Y to wyjscie
        loss_rpn = model_rpn.train_on_batch(X, Y)
        write_log(callback, ['rpn_cls_loss',
                             'rpn_reg_loss'], loss_rpn, train_step)

        # wygeneruj nowy Y korzystajac z poprzedniego modelu
        P_rpn = model_rpn.predict_on_batch(X)

        # przeksztalc rpn na obszary (bbox)
        R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(
        ), use_regr=True, overlap_thresh=0.7, max_boxes=300)

        # zamiana (x1,x2,y1,y2) na (x,y,w,h)
        X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

        # w przypadku braku X2 (zadnych obszarow) zakoncz ten epoch
        if X2 is None:
            rpn_accuracy_rpn_monitor.append(0)
            rpn_accuracy_for_epoch.append(0)
            continue

        # ustawianie pozytywne/negatywne przyklady (Y1 - lista klas do klasyfikacji)
        neg_samples = np.where(Y1[0, :, -1] == 1)
        pos_samples = np.where(Y1[0, :, -1] == 0)

        if len(neg_samples[0]) > 0:
            neg_samples = neg_samples[0]
        else:
            neg_samples = []

        if len(pos_samples) > 0:
            pos_samples = pos_samples[0]
        else:
            pos_samples = []

        # Zapisz pozytywne przyklady
        rpn_accuracy_rpn_monitor.append(len(pos_samples))
        rpn_accuracy_for_epoch.append((len(pos_samples)))

        if C.num_rois > 1:
            # Jesli pozytywnych przykladow jest mniej niz 2
            if len(pos_samples) < C.num_rois // 2:
                # Uzyj wszystkich przykladow
                selected_pos_samples = pos_samples.tolist()
            else:
                # Jesli wiecej, wylosuj 2 pozytywne przyklady
                selected_pos_samples = np.random.choice(
                    pos_samples, C.num_rois // 2, replace=False).tolist()
            try:
                # Wyszukaj negatywny przyklad
                selected_neg_samples = np.random.choice(
                    neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
            except ValueError:
                # W przpyadku braku negatywnych przykladow, skopiuj poprzedni negatywny przyklad
                try:
                    selected_neg_samples = np.random.choice(
                        neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()
                except:
                    # wyjatek - poprzedni negatywny przyklad to [[1 0 ]], wiec nie ma zadnych negatywnych przykladow
                    continue

            # Przekaz przyklady do sieci klasyfikatora
            sel_samples = selected_pos_samples + selected_neg_samples
        else:
            # w przypadku num_rois = 1 pozytywny lub negatywny przyklad jest losowany
            selected_pos_samples = pos_samples.tolist()
            selected_neg_samples = neg_samples.tolist()
            if np.random.randint(0, 2):
                sel_samples = random.choice(neg_samples)
            else:
                sel_samples = random.choice(pos_samples)

        # trening sieci klasyfikatora - dane obrazka, wsp wybranych przykladow, mapowanie wspolczynnikow
        loss_class = model_classifier.train_on_batch(
            [X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])
        write_log(callback, ['detection_cls_loss', 'detection_reg_loss',
                             'detection_acc'], loss_class, train_step)
        train_step += 1

        # Zapisanie strat
        losses[iter_num, 0] = loss_rpn[1]
        losses[iter_num, 1] = loss_rpn[2]

        losses[iter_num, 2] = loss_class[1]
        losses[iter_num, 3] = loss_class[2]
        losses[iter_num, 4] = loss_class[3]

        iter_num += 1
        export_counter += 1

        # zapisanie poszczegolnych strat do pliku
        with open(path + "/rpn_loss.csv", "a") as f:
            print(train_step, np.mean(losses[:iter_num, 0]), np.mean(losses[:iter_num, 1]),
                  np.mean(losses[:iter_num, 2]), np.mean(losses[:iter_num, 3]),
                  np.mean(losses[:iter_num, 0]) + np.mean(losses[:iter_num, 1]) + np.mean(
                      losses[:iter_num, 2]) + np.mean(losses[:iter_num, 3]),
                  sep=',', file=f)
        progbar.update(iter_num,
                       [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                        ('detector_cls', np.mean(losses[:iter_num, 2])),
                        ('detector_regr', np.mean(losses[:iter_num, 3]))])

        # Do backupu na Google Drive
        if export_counter == 50:
            try:
                shutil.copy(path + "/rpn_loss.csv", "/content/drive/My Drive/pracaMgr/rpn_loss.csv")
                export_counter = 0
            except Exception as e:
                print("Saving rpn_loss.csv was not possible")
                print(e)

        if iter_num == epoch_length:
            loss_rpn_cls = np.mean(losses[:, 0])
            loss_rpn_regr = np.mean(losses[:, 1])
            loss_class_cls = np.mean(losses[:, 2])
            loss_class_regr = np.mean(losses[:, 3])
            class_acc = np.mean(losses[:, 4])

            mean_overlapping_bboxes = float(
                sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
            rpn_accuracy_for_epoch = []

            if C.verbose:
                print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                    mean_overlapping_bboxes))
                print(
                    'Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                print('Loss RPN regression: {}'.format(loss_rpn_regr))
                print('Loss Detector classifier: {}'.format(loss_class_cls))
                print('Loss Detector regression: {}'.format(loss_class_regr))
                print('Elapsed time: {}'.format(time.time() - start_time))

            curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
            print("Current loss value: ", format(curr_loss))
            with open(path + "/losses_values.csv", "a") as f:
                print(epoch_num + 1, format(curr_loss), format(loss_rpn_regr), format(loss_rpn_cls),
                      format(time.time() - start_time), sep=',', file=f)
            iter_num = 0
            start_time = time.time()

            try:
                shutil.copy(path + "/losses_values.csv", "/content/drive/My Drive/pracaMgr/losses_values.csv")
            except Exception as e:
                print("Saving losses_values was not possible")
                print(e)

            write_log(callback,
                      ['Elapsed_time', 'mean_overlapping_bboxes', 'mean_rpn_cls_loss', 'mean_rpn_reg_loss',
                       'mean_detection_cls_loss', 'mean_detection_reg_loss', 'mean_detection_acc', 'total_loss'],
                      [time.time() - start_time, mean_overlapping_bboxes, loss_rpn_cls, loss_rpn_regr,
                       loss_class_cls, loss_class_regr, class_acc, curr_loss],
                      epoch_num)

            if curr_loss < best_loss:
                # Backupy od razu zapisywane sa na dysk
                filename = "weights" + str(curr_loss)[0:6] + ".hdf5"
                if C.verbose:
                    print('Total loss decreased from {} to {}, saving weights'.format(
                        best_loss, curr_loss))
                model_all.save_weights(C.model_path)
                try:
                    shutil.copy(path + "/weights.hdf5", "/content/drive/My Drive/pracaMgr/Weights/" + filename)
                except Exception as e:
                    print("Saving was not possible, sorry")
                    print(e)
                try:
                    os.remove(
                        "/content/drive/My Drive/pracaMgr/Weights/weights" + str(best_loss)[0:6] + ".hdf5")
                except OSError as e:
                    print("File removing was not possible")
                    print(e)
                best_loss = curr_loss

                if platform == "linux" or platform == "linux2":
                    if not os.path.exists("/content/drive/My Drive/pracaMgr/Weights/config.pickle"):
                        if os.path.exists("/content/config.pickle"):
                            shutil.copy("/content/config.pickle",
                                        "/content/drive/My Drive/pracaMgr/Weights/config.pickle")
                        elif os.path.exists("/content/pracaMgr/config.pickle"):
                            shutil.copy("/content/pracaMgr/config.pickle",
                                        "/content/drive/My Drive/pracaMgr/Weights/config.pickle")
                        elif os.path.exists("/content/pracaMgr/input/config.pickle"):
                            shutil.copy("/content/pracaMgr/input/config.pickle",
                                        "/content/drive/My Drive/pracaMgr/Weights/config.pickle")

            break

print('Training complete, exiting.')
