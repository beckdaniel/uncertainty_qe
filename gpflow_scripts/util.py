import numpy as np
import GPflow
import sklearn.preprocessing as pp
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from scipy.stats import pearsonr as pearson
import json


def read_data(feats_file, labels_file, size=None):
    """
    Read features and labels file, prune by size
    and output a numpy array where the last column
    is the label.
    """
    feats = np.loadtxt(feats_file)
    labels = np.loadtxt(labels_file, ndmin=2)
    if size:
        feats = feats[:size, :]
        labels = labels[:size, :]
    return np.concatenate((feats, labels), axis=1)


def normalize_train_data(train_data, hter=False):
    """
    Mean-normalize the features and turn the labels
    into time per word. We assume the second feature
    is the target sentence length. We also
    return the normalizer so it can be used for
    unseen data.
    """
    feats = train_data[:, :-1]
    labels = train_data[:, -1]
    if hter:
        labels_pw = labels
    else:
        labels_pw = labels / feats[:, 1]
    scaler = pp.StandardScaler()
    scaler.fit(feats)
    norm_feats = scaler.transform(feats)
    return np.concatenate((norm_feats, labels_pw[:, None]), axis=1), scaler


def normalize_test_data(test_data, scaler):
    """
    Mean-normalize the test features using the scaler 
    trained on the train features. Also turn the test labels
    into time per word.
    """
    feats = test_data[:, :-1]
    labels = test_data[:, -1]
    labels_pw = labels / feats[:, 1]
    norm_feats = scaler.transform(feats)
    return np.concatenate((norm_feats, labels_pw[:, None]), axis=1)


def train_gp_model(train_data, kernel='rbf', warp=None, ard=False, 
                   params_file=None, initial_y=0, likelihood='gaussian'):
    """
    Train a GP model with some training data.
    A model has:
    - A kernel (rbf, mat32, mat52)
    - ISO or ARD (F, T)
    - A warping function (t1, t2, t3, log, None)
      If None, we train a standard GP
    """
    train_feats = train_data[:, :-1]
    train_labels = train_data[:, -1:]

    # We first build the kernel
    if kernel == 'rbf':
        k = GPflow.kernels.RBF(17, ARD=ard)
    elif kernel == 'mat32':
        k = GPflow.kernels.Matern32(17, ARD=ard)
    elif kernel == 'mat52':
        k = GPflow.kernels.Matern52(17, ARD=ard)

    # The likelihood
    if likelihood == 'gaussian':
        ll = GPflow.likelihoods.Gaussian()
    elif likelihood == 'student':
        ll = GPflow.likelihoods.StudentT(deg_free=4.0)
        #ll.scale = 100.0

    # Now we build the warping function
    if warp == 'tanh1':
        w = GPflow.warping_functions.TanhFunction(n_terms=1)
        w.c = -1.0
    elif warp == 'tanh2':
        w = GPflow.warping_functions.TanhFunction(n_terms=2)
        w.c = [-1.0, -2.0]
    elif warp == 'tanh3':
        w = GPflow.warping_functions.TanhFunction(n_terms=3)
        w.c = [-1.0, -2.0, -3.0]
    elif warp == 'log':
        w = GPflow.warping_functions.LogFunction()

    if warp is not None and 'tanh' in warp:
        #w.a.transform = GPflow.transforms.Exp()
        #w.b.transform = GPflow.transforms.Exp()
        #w.d.transform = GPflow.transforms.Exp()
        w.d.fixed = True


    # Finally we instantiate the model
    if warp is None:
        gp = GPflow.vgp.VGP(train_feats, train_labels, k, ll)
    else:
        gp = GPflow.warped_gp.WarpedGP(train_feats, train_labels, k, warp=w, likelihood=ll)
        gp.median = True

    # TODO: initialize models with previous runs

    # Now we optimize
    #gp.optimize_restarts(num_restarts=10, max_iters=200, robust=True)

    gp.optimize(max_iters=100)
    # Sometimes we get warp hypers == 0 and this result in errors
    # in the prediction. We clip the values to a small value here
    if warp is not None and warp != 'log':
        clip_hypers(gp)
    return gp


def clip_hypers(gp):
    """
    Sometimes we get warp hypers == 0 and this result in errors
    in the prediction due to positive transforms. Note we don't
    need to do that to gp.c because it doesn't have a transform.
    """
    TOL = 1e-8
    gp.warp.a.set_state([[TOL if elem < TOL else elem for elem in gp.warp.a._array[0]]])
    gp.warp.b.set_state([[TOL if elem < TOL else elem for elem in gp.warp.b._array[0]]])
    gp.warp.d.set_state([TOL if elem < TOL else elem for elem in gp.warp.d._array])


def get_metrics(model, test_data):
    """
    Get predictions and evaluate.
    """
    feats = test_data[:, :-1]
    gold_labels = test_data[:, -1]
    preds = model.predict_y(feats)
    preds_mean = preds[0].flatten()
    preds_var = preds[1]
    #print preds_mean[:10]
    #print gold_labels[:10]
    mae = MAE(preds_mean, gold_labels)
    rmse = np.sqrt(MSE(preds_mean, gold_labels))
    prs = pearson(preds_mean, gold_labels)
    nlpd = - np.mean(model.predict_density(feats, gold_labels[:, None]))
    return mae, rmse, prs, nlpd


def save_parameters(gp, target):
    """
    Save the parameters of a GP to a json file.
    """
    pdict = {}
    pdict['likelihood'] = gp.likelihood.get_free_state()[0]
    pdict['kern_variance'] = gp.kern.variance.get_free_state()[0]
    pdict['kern_lengthscale'] = list(gp.kern.lengthscales.get_free_state())
    pdict['log_likelihood'] = gp._objective(gp.get_free_state())[0][0]
    #pdict = {n:list(gp[n].flatten()) for n in gp.parameter_names()}
    with open(target, 'w') as f:
        json.dump(pdict, f)


def save_gradients(gp, target):
    """
    Save the gradients of a GP.
    """
    np.savetxt(target, gp.gradient)


def save_metrics(metrics, target):
    with open(target, 'w') as f:
        f.write(str(metrics[0]) + '\n')
        f.write(str(metrics[1]) + '\n')
        f.write(str(metrics[2][0]) + '\n')
        f.write(str(metrics[2][1]) + '\n')
        f.write(str(metrics[3]) + '\n')


def save_cautious_curves(model, test_data, target, median=False):
    """
    Sort predictions by variance and calculate
    metrics on the top X% most confident ones,
    generating a curve on X.
    """
    feats = test_data[:, :-1]
    gold_labels = test_data[:, -1]
    if median: # should only be used for Warped GPs
        preds = model.predict_y(feats, median=True)
    else:
        preds = model.predict_y(feats)
    preds = zip(preds[0].flatten(), preds[1].flatten(), gold_labels)
    preds.sort(key=lambda x: x[0])
    preds = np.array(preds)
    metric_vals = []
    #import pprint; pprint.pprint(preds)
    for i in xrange(1, len(preds) + 1):
        sub_preds = preds[:i, 0]
        sub_gold = preds[:i, 2]
        mae = MAE(sub_preds, sub_gold)
        rmse = np.sqrt(MSE(sub_preds, sub_gold))
        prs = pearson(sub_preds, sub_gold)
        metric_vals.append([mae, rmse, prs[0], prs[1]])
    np.savetxt(target, metric_vals, fmt='%.4f')


def save_predictions(model, test_data, target, median=False):
    feats = test_data[:, :-1]
    gold_labels = test_data[:, -1]
    if median: # should only be used for Warped GPs
        preds = model.predict_y(feats, median=True)
    else:
        preds = model.predict_y(feats)
    preds = zip(preds[0].flatten(), preds[1].flatten(), gold_labels)
    preds.sort(key=lambda x: x[1])
    preds = np.array(preds)
    np.savetxt(target, preds, fmt='%.4f')


def load_parameters(gp, target):
    """
    Load the parameters of a GP from a json file.
    """
    with open(target) as f:
        pdict = json.load(f)
    gp.likelihood.set_state(pdict['likelihood'])
    gp.kern.variance.set_state(pdict['kern_variance'])
    gp.kern.lengthscales.set_state(pdict['kern_lengthscale'])
    #for p in pdict:
    #    if p == 'warp_tanh.psi':
    #        gp[p] = np.array(pdict[p]).reshape(3, 3)
    #    else:
    #        gp[p] = pdict[p]



if __name__ == "__main__":
    import sys
    FEATS_FILE = sys.argv[1]
    LABELS_FILE = sys.argv[2]
    SIZE = int(sys.argv[3])
    SPLIT = int(sys.argv[4])
    kernel = sys.argv[5]
    ard = sys.argv[7]
    if ard == 'False':
        ard = False
    else:
        ard = True
    warp = sys.argv[6]
    if warp == "None":
        warp = None
    ll = sys.argv[8]
    data = read_data(FEATS_FILE, LABELS_FILE, size=SIZE)
    train_data = data[:SPLIT, :]
    test_data = data[SPLIT:, :]
    #train_data, scaler = normalize_train_data(train_data, hter=True)
    train_data, scaler = normalize_train_data(train_data, hter=False)
    test_data = normalize_test_data(test_data, scaler)

    gp = train_gp_model(train_data, kernel, warp, ard, likelihood=ll)
    #print gp
    print gp.likelihood
    import ipdb; ipdb.set_trace()
    #gp.checkgrad()
    rmse, ps, nlpd = get_metrics(gp, test_data)
    import ipdb; ipdb.set_trace()
    print "RMSE:\t\t%.4f" % rmse
    print "Pearsons:\t%.4f\t%.4f" % ps
    print "NLPD:\t\t%.4f" % nlpd
    #save_parameters(gp, '../saved_models/' + model)
    
