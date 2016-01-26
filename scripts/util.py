import numpy as np
import GPy
import sklearn.preprocessing as pp
from sklearn.metrics import mean_squared_error as MSE
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


def normalize_train_data(train_data):
    """
    Mean-normalize the features and turn the labels
    into time per word. We assume the second feature
    is the target sentence length. We also
    return the normalizer so it can be used for
    unseen data.
    """
    feats = train_data[:, :-1]
    labels = train_data[:, -1]
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


def train_gp_model(train_data, model):
    """
    Train a GP model with some training data.
    """
    train_feats = train_data[:, :-1]
    train_labels = train_data[:, -1:]
    if model == 'gp_rbf_iso':
        gp = GPy.models.GPRegression(train_feats, train_labels)
    elif model == 'gp_rbf_ard':
        rbf = GPy.kern.RBF(17, ARD=True)
        gp = GPy.models.GPRegression(train_feats, train_labels, kernel=rbf)
    if model == 'gp_mat32_ard':
        mat32 = GPy.kern.Matern32(17, ARD=True)
        gp = GPy.models.GPRegression(train_feats, train_labels, kernel=mat32)
    if model == 'wgp_mat32_ard':
        mat32 = GPy.kern.Matern32(17, ARD=True)
        warp = GPy.util.warping_functions.TanhWarpingFunction_d(initial_y=0)
        gp = GPy.models.WarpedGP(train_feats, train_labels, kernel=mat32, warping_function=warp)
    if model == 'wgp_rbf_iso':
        rbf = GPy.kern.RBF(17)
        warp = GPy.util.warping_functions.TanhWarpingFunction_d(initial_y=0)
        gp = GPy.models.WarpedGP(train_feats, train_labels, kernel=rbf, warping_function=warp)
    gp.optimize_restarts(num_restarts=5, max_iters=100, robust=True)

    #gp.optimize()
    return gp


def get_metrics(model, test_data):
    """
    Get predictions and evaluate.
    """
    feats = test_data[:, :-1]
    gold_labels = test_data[:, -1]
    preds = model.predict(feats)
    preds_mean = preds[0].flatten()
    preds_var = preds[1]
    #print preds_mean[:10]
    #print gold_labels[:10]
    rmse = np.sqrt(MSE(preds_mean, gold_labels))
    prs = pearson(preds_mean, gold_labels)
    nlpd = - np.mean(model.log_predictive_density(feats, gold_labels[:, None]))
    return rmse, prs, nlpd


def save_parameters(gp, target):
    """
    Save the parameters of a GP to a json file.
    """
    pdict = {n:list(gp[n].flatten()) for n in gp.parameter_names()}
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
        f.write(str(metrics[1][0]) + '\n')
        f.write(str(metrics[1][1]) + '\n')
        f.write(str(metrics[2]) + '\n')


def load_parameters(gp, target):
    """
    Load the parameters of a GP from a json file.
    """
    with open(target) as f:
        pdict = json.load(f)
    for p in pdict:
        if p == 'warp_tanh.psi':
            gp[p] = np.array(pdict[p]).reshape(3, 3)
        else:
            gp[p] = pdict[p]



if __name__ == "__main__":
    import sys
    FEATS_FILE = sys.argv[1]
    LABELS_FILE = sys.argv[2]
    SIZE = int(sys.argv[3])
    SPLIT = int(sys.argv[4])
    data = read_data(FEATS_FILE, LABELS_FILE, size=SIZE)
    train_data = data[:SPLIT, :]
    test_data = data[SPLIT:, :]
    train_data, scaler = normalize_train_data(train_data)
    test_data = normalize_test_data(test_data, scaler)

    model = 'gp_rbf_iso'
    #model = 'gp_rbf_ard'
    #model = 'gp_mat32_ard'
    #model = 'wgp_mat32_ard'
    model = 'wgp_rbf_iso'
    gp = train_gp_model(train_data, model)
    print gp
    rmse, ps, nlpd = get_metrics(gp, test_data)
    print "RMSE:\t\t%.4f" % rmse
    print "Pearsons:\t%.4f\t%.4f" % ps
    print "NLPD:\t\t%.4f" % nlpd
    save_parameters(gp, '../saved_models/' + model)
    import ipdb; ipdb.set_trace()
