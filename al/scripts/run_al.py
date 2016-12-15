import numpy as np
import sys
import os
import GPy
from sklearn import preprocessing as pp
import gc
import util


def train_al_and_report(model_name, kernel, warp, ard):
    dataset_dir = os.path.join(MODEL_DIR, DATASET)
    try: 
        os.makedirs(dataset_dir)
    except OSError:
        print "skipping output folder"
    for fold in xrange(1):
        fold_dir = os.path.join(SPLIT_DIR, DATASET, str(fold))
        train_data = np.loadtxt(os.path.join(fold_dir, 'train'))
        test_data = np.loadtxt(os.path.join(fold_dir, 'test'))
        params_file = None
        output_dir = os.path.join(dataset_dir, str(fold))
        try: 
            os.makedirs(output_dir)
        except OSError:
            print "skipping output folder"
        if ard:
            iso_dir = output_dir.replace('True', 'False')
            params_file = os.path.join(iso_dir, 'params')
            
        # Split train into train and pool
        pool_data = train_data[50:]
        train_data = train_data[:50]

        metrics_list = []
        while True:
            # Train gp
            gp = util.train_gp_model(train_data, kernel, warp, ard, params_file)
            
            # Get metrics on test
            metrics = util.get_metrics(gp, test_data)
            metrics_list.append([metrics[0], metrics[1], metrics[2][0],
                                 metrics[2][1], metrics[3]])
            
            # Predict pool and select instance in pool with higher variance
            new_instance, new_i = util.query(gp, pool_data)

            # Update train and pool
            train_data = np.append(train_data, [new_instance], axis=0)
            pool_data = np.delete(pool_data, (new_i), axis=0)
            
            #if pool_data.shape[0] == 0:
            #    break
            if train_data.shape[0] == 500:
                break

            print pool_data.shape
            gc.collect(2)

        # Final metrics on full train set (sanity check)
        gp = util.train_gp_model(train_data, kernel, warp, ard, params_file)
        metrics = util.get_metrics(gp, test_data)
        metrics_list.append([metrics[0], metrics[1], metrics[2][0],
                             metrics[2][1], metrics[3]])

        util.save_metrics_list(metrics_list, os.path.join(output_dir, 'metrics'))
        

DATASET = sys.argv[1]
KERNEL = sys.argv[2]
WARP = sys.argv[3]
ARD = sys.argv[4]
MODEL_NAME = '_'.join([KERNEL, WARP, ARD])
SPLIT_DIR = os.path.join('..','..','splits')

if ARD == 'False':
    ARD = False
else:
    ARD = True
if WARP == "None":
    WARP = None

MODEL_DIR = os.path.join('..', 'models', MODEL_NAME)
train_al_and_report(MODEL_NAME, KERNEL, WARP, ARD)
