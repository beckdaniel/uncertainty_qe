import numpy as np
import GPy
import util
import sys
import os
from sklearn import cross_validation, preprocessing


DATASETS = ['eamt11_fr-en', 'eamt11_en-es', 'wmt14_en-es']
DATASET = sys.argv[1]
MODEL = sys.argv[2]

if DATASET not in DATASETS:
    print DATASETS
    sys.exit(1)

# READ DATA
dataset_dir = os.path.join('..', 'data', DATASET)
feats_file = os.path.join(dataset_dir, 'feats.17')
labels_file = os.path.join(dataset_dir, 'time')
data = util.read_data(feats_file, labels_file)

# SHUFFLE DATA
np.random.seed(1000)
np.random.shuffle(data)

# SPLIT TRAIN/TEST
fold_indices = cross_validation.KFold(data.shape[0], n_folds=2)
for index in fold_indices:
    print index[0].shape

# SAVED MODELS STRUCTURE
model_dir = os.path.join('..', 'models', DATASET, MODEL)
try:
    os.makedirs(model_dir)
except OSError:
    print "model_dir already exists, skipping"
for fold, index in enumerate(fold_indices):
    fold_file = os.path.join(model_dir, str(fold) + '.params')
    grad_file = os.path.join(model_dir, str(fold) + '.grads')
    metrics_file = os.path.join(model_dir, str(fold) + '.metrics')
    train_data = data[index[0]]
    test_data = data[index[1]]
    train_data, scaler = util.normalize_train_data(train_data)
    test_data = util.normalize_test_data(test_data, scaler)
    gp = util.train_gp_model(train_data, MODEL)
    metrics = util.get_metrics(gp, test_data)
    util.save_parameters(gp, fold_file)
    util.save_gradients(gp, grad_file)
    util.save_metrics(metrics, metrics_file)
