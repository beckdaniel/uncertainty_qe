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
        dataset_dir = os.path.join(SPLIT_DIR, DATASET)
        try:
            os.makedirs(dataset_dir)
        except OSError:
            print "skipping folder creation"

        # SPLIT TRAIN/TEST AND SAVE
        fold_indices = cross_validation.KFold(data.shape[0], n_folds=10)
        for fold, index in enumerate(fold_indices):
            print index[0].shape
            train_data = data[index[0]]
            test_data = data[index[0]]
            train_data, scaler = util.normalize_train_data(train_data)
            test_data = util.normalize_test_data(test_data, scaler)

            fold_dir = os.path.join(dataset_dir, str(fold))
            try:
                os.makedirs(fold_dir)
            except OSError:
                print "skipping fold dir"
            np.savetxt(os.path.join(fold_dir, 'train'), train_data, fmt="%.5f")
            np.savetxt(os.path.join(fold_dir, 'test'), test_data, fmt="%.5f")


def train_and_report(model):
    for DATASET in DATASETS:
        dataset_dir = os.path.join(MODEL_DIR, DATASET)
        try: 
            os.makedirs(dataset_dir)
        except OSError:
            print "skipping output folder"
        for fold in xrange(10):
            fold_dir = os.path.join(SPLIT_DIR, DATASET, str(fold))
            train_data = np.loadtxt(os.path.join(fold_dir, 'train'))
            test_data = np.loadtxt(os.path.join(fold_dir, 'train'))
            gp = util.train_gp_model(train_data, model)
            metrics = util.get_metrics(gp, test_data)
            output_dir = os.path.join(dataset_dir, str(fold))
            try: 
                os.makedirs(output_dir)
            except OSError:
                print "skipping output folder"
            util.save_parameters(gp, os.path.join(output_dir, 'params'))
            util.save_metrics(metrics, os.path.join(output_dir, 'metrics'))
            util.save_gradients(gp, os.path.join(output_dir, 'grads'))
                
    

###########################    

# Default is to run on all datasets, this can be changed by
# using a different list with fewer elements.
#DATASETS = ['eamt11_fr-en', 'eamt11_en-es', 'wmt14_en-es']
DATASETS = ['wmt14_en-es']
SPLIT_DIR = os.path.join('..','splits')

# Generate the splits. Each split has mean-normalized features and
# pe-time per word in target segment.
split_all_data()

# The simplest model of all: GP RBF Isotropic
MODEL_DIR = os.path.join('..', 'models', 'gp_rbf_iso')
train_and_report('gp_rbf_iso')
