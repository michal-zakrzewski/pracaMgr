import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.data import imread
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

train = os.listdir('../input/train')
print(len(train))

test = os.listdir('../input/test')
print(len(test))

submission = pd.read_csv('../input/sample_submission.csv')
submission.head()
