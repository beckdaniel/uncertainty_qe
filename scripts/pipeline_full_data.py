import numpy as np
import GPy
import util
import sys
import os
from sklearn import cross_validation, preprocessing
import gc


def split_all_data():
    # READ DATA
    dataset_dir = os.path.join('..', 'data', DATASET)
    feats_file = os.path.join(dataset_dir, 'feats.17')
    labels_file = os.path.join(dataset_dir, 'time')
    source_file = os.path.join(dataset_dir, 'source')
    target_file = os.path.join(dataset_dir, 'target')
    pe_file = os.path.join(dataset_dir, 'target_postedited')
    #data = util.read_data(feats_file, labels_file)
    
    feats = np.loadtxt(feats_file, dtype=object)
    labels = np.loadtxt(labels_file, dtype=object, ndmin=2)
    src = np.loadtxt(source_file, dtype=object, delimiter='\t', ndmin=2)
    tgt = np.loadtxt(target_file, dtype=object, delimiter='\t', ndmin=2)
    pe = np.loadtxt(pe_file, dtype=object, delimiter='\t', ndmin=2)

    data = np.concatenate((feats, labels, src, tgt, pe), axis=1)
    

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
        train = data[index[0]]
        test = data[index[1]]
        train_data = np.array(train[:, :18], dtype=float)
        test_data = np.array(test[:, :18], dtype=float)
        train_data, scaler = util.normalize_train_data(train_data)
        test_data = util.normalize_test_data(test_data, scaler)

        fold_dir = os.path.join(dataset_dir, str(fold))
        try:
            os.makedirs(fold_dir)
        except OSError:
            print "skipping fold dir"
        np.savetxt(os.path.join(fold_dir, 'train'), train_data, fmt="%.5f")
        np.savetxt(os.path.join(fold_dir, 'test'), test_data, fmt="%.5f")
        np.savetxt(os.path.join(fold_dir, 'train_src'), train[:, 18], fmt="%s")
        np.savetxt(os.path.join(fold_dir, 'test_src'), test[:, 18], fmt="%s")
        np.savetxt(os.path.join(fold_dir, 'train_tgt'), train[:, 19], fmt="%s")
        np.savetxt(os.path.join(fold_dir, 'test_tgt'), test[:, 19], fmt="%s")
        np.savetxt(os.path.join(fold_dir, 'train_pe'), train[:, 20], fmt="%s")
        np.savetxt(os.path.join(fold_dir, 'test_pe'), test[:, 20], fmt="%s")




def train_and_report(model_name, kernel, warp, ard):
    dataset_dir = os.path.join(MODEL_DIR, DATASET)
    try: 
        os.makedirs(dataset_dir)
    except OSError:
        print "skipping output folder"
    for fold in xrange(10):
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
        gp = util.train_gp_model(train_data, kernel, warp, ard, params_file)
        util.save_parameters(gp, os.path.join(output_dir, 'params'))
        util.save_gradients(gp, os.path.join(output_dir, 'grads'))
        metrics = util.get_metrics(gp, test_data)



        util.save_metrics(metrics, os.path.join(output_dir, 'metrics'))
        util.save_cautious_curves(gp, test_data, os.path.join(output_dir, 'curves'))
        util.save_predictions(gp, test_data, os.path.join(output_dir, 'preds'))

        asym_metrics = util.get_asym_metrics(gp, test_data)
        util.save_asym_metrics(asym_metrics, os.path.join(output_dir, 'asym_metrics'))
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
    train_and_report(MODEL_NAME, KERNEL, WARP, ARD)
