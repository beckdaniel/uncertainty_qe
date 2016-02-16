import numpy as np
import GPy
import util
import sys
import os
from sklearn import cross_validation, preprocessing


def split_all_data():
    for DATASET in DATASETS:
        # READ DATA
        dataset_dir = os.path.join('..', 'data', DATASET)
        feats_file = os.path.join(dataset_dir, 'feats.17')
        labels_file = os.path.join(dataset_dir, 'time')
        data = util.read_data(feats_file, labels_file)

        # SHUFFLE DATA
        np.random.seed(1000)
        np.random.shuffle(data)

        # BUILD FOLDER STRUCTURE
        dataset_dir = os.path.join('..','splits', DATASET)
        try:
            os.makedirs(dataset_dir)
        except OSError:
            print "skipping folder creation"

        # SPLIT TRAIN/TEST AND SAVE
        fold_indices = cross_validation.KFold(data.shape[0], n_folds=10)
        for i, index in enumerate(fold_indices):
            print index[0].shape
            train_data = data[index[0]]
            test_data = data[index[0]]
            train_data, scaler = util.normalize_train_data(train_data)
            test_data = util.normalize_test_data(test_data, scaler)

            fold_dir = os.path.join(dataset_dir, str(i))
            try:
                os.makedirs(fold_dir)
            except OSError:
                print "skipping fold dir"
            np.savetxt(os.path.join(fold_dir, 'train'), train_data, fmt="%.5f")
            np.savetxt(os.path.join(fold_dir, 'test'), test_data, fmt="%.5f")
    
###########################    

DATASETS = ['eamt11_fr-en', 'eamt11_en-es', 'wmt14_en-es']

# Generate the splits. Each split has mean-normalized features and
# pe-time per word in target segment.
split_all_data()
