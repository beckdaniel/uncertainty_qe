import numpy as np
import GPy
import util
import sys
import os
from sklearn import cross_validation, preprocessing
import gc


def eval_and_report(model_name, kernel, warp, ard):
    dataset_dir = os.path.join(MODEL_DIR, DATASET)
    for fold in xrange(10):
        fold_dir = os.path.join(SPLIT_DIR, DATASET, str(fold))
        train_data = np.loadtxt(os.path.join(fold_dir, 'train'))
        test_data = np.loadtxt(os.path.join(fold_dir, 'test'))
        model_dir = os.path.join(dataset_dir, str(fold))
        params_file = os.path.join(model_dir, 'params')

        gp = util.train_gp_model(train_data, kernel, warp, ard, 
                                 params_file=params_file, preload=True)
        #metrics = util.get_metrics(gp, test_data)
        #asym_metrics = util.get_asym_metrics(gp, test_data)
        #util.save_asym_metrics(asym_metrics, os.path.join(model_dir, 'asym_metrics'))
        #rec_metrics = util.get_rec_metrics(gp, test_data)
        #util.save_rec_metrics(rec_metrics, os.path.join(model_dir, 'rec_metrics'))
        linex_metrics = util.get_linex_metrics(gp, test_data)
        util.save_linex_metrics(linex_metrics, os.path.join(model_dir, 'linex_metrics'))
        gc.collect(2) # buggy GPy has allocation cycles...
                
    
###########################    

DATASET = sys.argv[1]
KERNEL = sys.argv[2]
WARP = sys.argv[3]
ARD = sys.argv[4]
MODEL_NAME = '_'.join([KERNEL, WARP, ARD])

if ARD == 'False':
    ARD = False
else:
    ARD = True
if WARP == "None":
    WARP = None

SPLIT_DIR = os.path.join('..','splits')

# Generate the splits. Each split has mean-normalized features and
# pe-time per word in target segment.
if KERNEL == 'split':
    split_all_data()
else:
    MODEL_DIR = os.path.join('..', 'models', MODEL_NAME)
    eval_and_report(MODEL_NAME, KERNEL, WARP, ARD)
