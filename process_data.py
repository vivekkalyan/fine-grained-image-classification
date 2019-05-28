import os

import numpy as np
import pandas as pd

from scipy.io import loadmat

data_dir = "data"
train_file = "devkit/cars_train_annos.mat"
test_file = "devkit/cars_test_annos_withlabels.mat"

train = loadmat(os.path.join(data_dir, train_file))["annotations"][0]
test = loadmat(os.path.join(data_dir, test_file))["annotations"][0]

def to_pandas(mat):
    dataset = []
    for row in mat:
        # row
        # bbox_x1: array([[39]], dtype=uint8),
        # bbox_y1: array([[116]], dtype=uint8),
        # bbox_x2: array([[569]], dtype=uint16),
        # bbox_y2: array([[375]], dtype=uint16),
        # class: array([[14]], dtype=uint8),
        # fname: array(['00001.jpg'], dtype='<U9'))
        values = [r[0][0] for r in row if (np.issubdtype(r.dtype, np.integer))]
        dataset.append([row[-1][0], *values])
    columns=['filename','bbox_x1','bbox_y1','bbox_x2','bbox_y2']
    assert(len(dataset[0]) == 6)
    columns.append('class_id')
    df = pd.DataFrame(dataset, columns=columns)
    return df

train_df = to_pandas(train)
test_df = to_pandas(test)
