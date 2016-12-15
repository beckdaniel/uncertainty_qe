import numpy as np
import os
import sys
import itertools as it
import json

DATASET = sys.argv[1]
PRED = sys.argv[2]
MODELS_DIR = os.path.join('..', 'models')
KERNELS = ['rbf', 'mat32', 'mat52']
#LIKELIHOODS = ['None', 'log', 'tanh1', 'tanh2', 'tanh3']
ARD = ['False', 'True']
#KERNELS = ['mat32']
ARD = ['False']
ARD = ['True']
if PRED == 'mean' or PRED == 'asym' or PRED == 'recip' or PRED == 'linex':
    LIKELIHOODS = ['None', 'log', 'tanh1', 'tanh2', 'tanh3']#, 'logtanh1', 'logtanh2', 'logtanh3']
    LIKELIHOODS = ['None']
    LIKELIHOODS = ['log']
    LIKELIHOODS = ['tanh1']
    LIKELIHOODS = ['tanh2']
    LIKELIHOODS = ['tanh3']
    combs = it.product(KERNELS, LIKELIHOODS, ARD)
elif PRED == 'median':
    LIKELIHOODS = ['log', 'tanh1', 'tanh2', 'tanh3']#, 'logtanh1', 'logtanh2', 'logtanh3']
    LIKELIHOODS = ['tanh3']
    combs = it.product(KERNELS, LIKELIHOODS, ARD, ['median'])
elif PRED == 'student':
    LIKELIHOODS = ['None']
    combs = it.product(KERNELS, LIKELIHOODS, ARD, ['student'])


CUTOFF = 5
CUTOFF2 = -1

def get_capacities(filename):
    metrics = np.loadtxt(filename)
    return np.mean(metrics[CUTOFF:CUTOFF2], axis=0)

if sys.argv[2] == 'asym':
    pass
    #print '\t'.join(['PESS', 'OPTIM'])
elif sys.argv[2] == 'recip':
    print '\t'.join(['%20s' % '', 'NAIVE', '\t\t', 'RECIP'])
    print '\t'.join(['%20s' % '', 'MAE', 'RMSE', 'Pearson', 'p', 'MAE', 'RMSE', 'Pearson', 'p'])
else:
    print '\t'.join(['%20s' % '', 'MAE', 'RMSE', 'Pearson', 'p', 'NLPD', 'cMAE', 'cRMSE', 'cPrs', 'cp'])
#for comb in it.product(KERNELS, LIKELIHOODS, ARD, PRED):
for comb in combs:
    model = '_'.join(comb)
    model_dir = os.path.join(MODELS_DIR, model, DATASET)
    all_metrics = []
    all_asyms = []
    all_recips = []
    all_linex = []
    for fold in xrange(10):
        try:
            metrics = np.loadtxt(os.path.join(model_dir, str(fold), 'metrics'))
            #print metrics
            capacities = get_capacities(os.path.join(model_dir, str(fold), 'curves'))
            #print capacities
            with open(os.path.join(model_dir, str(fold), 'params')) as f:
                params = json.load(f)
            log_likelihood = np.array([params['log_likelihood']])
            all_metrics.append(np.concatenate((metrics, capacities, log_likelihood), axis=0))
            asyms = np.loadtxt(os.path.join(model_dir, str(fold), 'asym_metrics'))
            #recips = np.loadtxt(os.path.join(model_dir, str(fold), 'rec_metrics'), dtype=object)
            linex = np.loadtxt(os.path.join(model_dir, str(fold), 'linex_metrics'))
            all_asyms.append(asyms)
            all_linex.append(linex)
            #all_recips.append([[float(e[0]), float(e[1])] for e in recips[1:]])
        except IOError:
            print "BLAH"
    if sys.argv[2] == 'asym':
        print model
        #print all_asyms
        mean_norm_asyms = (np.mean(all_asyms, axis=0)).T# / np.arange(1, 5.1, 0.5)[:,None]).T
        print '\t'.join(['PESS'] + ['%.4f' % e for e in mean_norm_asyms[0]])
        print '\t'.join(['OPTIM'] + ['%.4f' % e for e in mean_norm_asyms[1]])
    elif sys.argv[2] == 'linex':
        print model
        mean_norm_asyms = (np.mean(all_linex, axis=0)).T# / np.arange(2, 10.1, 1)[:,None]).T
        print '\t'.join(['PESS'] + ['%.4f' % e for e in mean_norm_asyms[0, :4]])
        print '\t'.join(['OPTIM'] + ['%.4f' % e for e in mean_norm_asyms[1, :4]])
    elif sys.argv[2] == 'recip':
        mean_recips = (np.mean(all_recips, axis=0))
        naive = mean_recips[:,0].flatten().T
        recip = mean_recips[:,1].flatten().T
        print '\t'.join(['%20s' % model[:20]] + ['%.4f' % e for e in np.concatenate((naive, recip))])
    else:
        print '\t'.join(['%20s' % model[:20]] + ['%.4f' % e if e < 10 else '%.1e' % e for e in np.mean(all_metrics, axis=0)])

    #print '\t'.join(['%20s' % model] + ['%.4f' % e if e < 10 else '%.1e' % e for e in np.std(all_metrics, axis=0)])
