import numpy as np
import GPy
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
                   params_file=None, initial_y=0, preload=False):
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
        k = GPy.kern.RBF(17, ARD=ard)
    elif kernel == 'mat32':
        k = GPy.kern.Matern32(17, ARD=ard)
    elif kernel == 'mat52':
        k = GPy.kern.Matern52(17, ARD=ard)

    # Now we build the warping function
    if warp == 'tanh1':
        w = GPy.util.warping_functions.TanhFunction(n_terms=1, initial_y=initial_y)
        w.psi[:,2] = -1.0
        w.d.constrain_fixed(1)
    elif warp == 'tanh2':
        w = GPy.util.warping_functions.TanhFunction(n_terms=2, initial_y=initial_y)
        w.psi[:,2] = [-1.0, -2.0]
        w.d.constrain_fixed(1)
    elif warp == 'tanh3':
        w = GPy.util.warping_functions.TanhFunction(n_terms=3, initial_y=initial_y)
        w.psi[:,2] = [-1.0, -2.0, -3.0]
        w.d.constrain_fixed(1)
    elif warp == 'logtanh1':
        w = GPy.util.warping_functions.LogTanhFunction(n_terms=1, initial_y=initial_y)
        w.psi[:,2] = -1.0
        w.d.constrain_fixed(1)
    elif warp == 'logtanh2':
        w = GPy.util.warping_functions.LogTanhFunction(n_terms=2, initial_y=initial_y)
        w.psi[:,2] = [-1.0, -2.0]
        w.d.constrain_fixed(1)
    elif warp == 'logtanh3':
        w = GPy.util.warping_functions.LogTanhFunction(n_terms=3, initial_y=initial_y)
        w.psi[:,2] = [-1.0, -2.0, -3.0]
        w.d.constrain_fixed(1)
    elif warp == 'log':
        w = GPy.util.warping_functions.LogFunction()

    # Finally we instantiate the model
    if warp is None:
        gp = GPy.models.GPRegression(train_feats, train_labels, kernel=k)
    else:
        #w.rate = 0.15
        gp = GPy.models.WarpedGP(train_feats, train_labels, kernel=k, warping_function=w)

    # Now we optimize
    # We use random restarts for isotropic models and
    # preloaded models for ARD ones.
    if params_file is not None:
        load_parameters(gp, params_file)
        if not preload: # we might as well just preload
            gp.optimize(max_iters=100)
    else:
        gp.optimize_restarts(num_restarts=10, max_iters=100, robust=True)
    return gp


def get_metrics(model, test_data, median=False):
    """
    Get predictions and evaluate.
    """
    feats = test_data[:, :-1]
    gold_labels = test_data[:, -1]
    if median and isinstance(model, GPy.models.WarpedGP): # should only be used for Warped GPs
        preds = model.predict(feats, median=True)
    else:
        preds = model.predict(feats)
    preds_mean = preds[0].flatten()
    preds_var = preds[1]
    #print preds_mean[:10]
    #print gold_labels[:10]
    mae = MAE(preds_mean, gold_labels)
    rmse = np.sqrt(MSE(preds_mean, gold_labels))
    prs = pearson(preds_mean, gold_labels)
    nlpd = - np.mean(model.log_predictive_density(feats, gold_labels[:, None]))
    pred_q = model.predict_quantiles(feats, quantiles=(25., 75.))[1].flatten()    
    return mae, rmse, prs, nlpd


def get_rec_metrics(model, test_data, median=False):
    """
    Get predictions and evaluate.
    """
    feats = test_data[:, :-1]
    gold_labels = 1 / test_data[:, -1]
    if median: # should only be used for Warped GPs
        preds = model.predict(feats, median=True)
    else:
        preds = model.predict(feats)
    preds_mean = preds[0].flatten()
    rec_preds = model.predict_reciprocal(feats).flatten()

    mae_naive = MAE(1/preds_mean, gold_labels)
    rmse_naive = np.sqrt(MSE(1/preds_mean, gold_labels))
    prs_naive = pearson(1/preds_mean, gold_labels)

    mae_rec = MAE(rec_preds, gold_labels)
    rmse_rec = np.sqrt(MSE(rec_preds, gold_labels))
    prs_rec = pearson(rec_preds, gold_labels)
    return mae_naive, rmse_naive, prs_naive, mae_rec, rmse_rec, prs_rec


def get_asym_metrics(model, test_data):
    feats = test_data[:, :-1]
    gold_labels = test_data[:, -1]
    pess_maes = []
    opt_maes = []
    for i in xrange(1, 10):
        q = (i / (1. + i)) * 100
        preds_q_low, preds_q_high = model.predict_quantiles(feats, quantiles=(100 - q, q))
        pess_mae = asym_mae(preds_q_high.flatten(), gold_labels, w=i)
        opt_mae = asym_mae(preds_q_low.flatten(), gold_labels, w=i, optimistic=True)
        pess_maes.append(pess_mae)
        opt_maes.append(opt_mae)
    return [pess_maes, opt_maes]


def asym_mae(preds, gold, w=3, optimistic=False):
    diffs = preds - gold
    if optimistic:
        zeros = diffs > 0
    else:
        zeros = diffs < 0
    weights = (zeros * (w-1)) + 1 #hacky
    return np.mean(np.abs(diffs) * weights)


def get_linex_metrics(model, test_data):
    feats = test_data[:, :-1]
    gold_labels = test_data[:, -1]
    pess_linexes = []
    opt_linexes = []    
    for w in np.arange(0.25, 2.01, 0.25):
        means, variances = model.predict(feats)
        preds_pess = means + ((w * variances) / 2.)
        preds_opt = means - ((w * variances) / 2.)
        pess_linex = linex_loss(preds_pess, gold_labels, w=-w)
        opt_linex = linex_loss(preds_opt, gold_labels, w=w)
        pess_linexes.append(pess_linex)
        opt_linexes.append(opt_linex)
        #break
    return [pess_linexes, opt_linexes]


def linex_loss(preds, gold, w=3):
    delta = (preds - gold).flatten()
    return np.mean(np.exp(delta * w) - (delta * w) - 1)


def save_asym_metrics(metrics, target):
    np.savetxt(target, np.array(metrics).T, fmt='%.4f')


def save_linex_metrics(metrics, target):
    np.savetxt(target, np.array(metrics).T, fmt='%.4f')


def save_rec_metrics(metrics, target):
    with open(target, 'w') as f:
        f.write('%5s\t%5s\n' % ('NAIVE', 'RECIP'))
        f.write('%.4f\t%.4f\n' % (metrics[0], metrics[3]))
        f.write('%.4f\t%.4f\n' % (metrics[1], metrics[4]))
        f.write('%.4f\t%.4f\n' % (metrics[2][0], metrics[5][0]))
        f.write('%.4f\t%.4f\n' % (metrics[2][1], metrics[5][1]))


def save_parameters(gp, target):
    """
    Save the parameters of a GP to a json file.
    """
    pdict = {n:list(gp[n].flatten()) for n in gp.parameter_names()}
    pdict['log_likelihood'] = gp.log_likelihood()
    with open(target, 'w') as f:
        json.dump(pdict, f)


def save_gradients(gp, target):
    """
    Save the gradients of a GP.
    """
    np.savetxt(target, gp.gradient)


def save_al_metrics(al_metrics, target):
    """
    Save the sequence of metrics from AL.
    """
    np.savetxt(target, np.array(al_metrics), fmt='%.4f')


def save_metrics(metrics, target):
    with open(target, 'w') as f:
        f.write(str(metrics[0]) + '\n')
        f.write(str(metrics[1]) + '\n')
        f.write(str(metrics[2][0]) + '\n')
        f.write(str(metrics[2][1]) + '\n')
        f.write(str(metrics[3]) + '\n')


def save_metrics_list(metrics, target):
    np.savetxt(target, np.array(metrics), fmt='%.4f')


def save_cautious_curves(model, test_data, target, median=False):
    """
    Sort predictions by variance and calculate
    metrics on the top X% most confident ones,
    generating a curve on X.
    """
    feats = test_data[:, :-1]
    gold_labels = test_data[:, -1]
    if median: # should only be used for Warped GPs
        preds = model.predict(feats, median=True)
    else:
        preds = model.predict(feats)
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
        preds = model.predict(feats, median=True)
    else:
        preds = model.predict(feats)
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
    for p in pdict:
        if p == 'warp_tanh.psi':
            gp[p] = np.array(pdict[p]).reshape(len(pdict[p]) / 3, 3)
        elif p != 'log_likelihood':
            gp[p] = pdict[p]


def dloss(ai, yi, yk):
    n = yk.shape[0]
    dot_ai = np.dot(ai.T, ai)
    sum_ai = np.sum(ai)
    dot_aiyi = np.dot(ai, yi)
    sum_yi = np.sum(yi, axis=0)

    num1 = n * dot_ai * yk
    num2 = (sum_ai ** 2) * yk
    num3 = sum_ai * dot_aiyi
    num4 = dot_ai * sum_yi
    num5 = dot_ai * yk
    num = num1 - num2 + num3 - num4 - num5
    
    denom1 = n * dot_aiyi
    denom2 = sum_ai * sum_yi
    denom3 = dot_aiyi
    denom = denom1 - denom2 - denom3

    return np.sum(num / denom)


def norm_a(a, mean):
    mu, sigma = (a.mean(), a.std())
    new_a = (a - mu) / sigma
    mu_y, sigma_y = (mean.mean(), mean.std())
    new_a *= sigma_y
    new_a += mu_y
    return new_a


def prs_loss(mean, cov, samples=1000, its=10):
    curr_a = np.copy(mean) # start with mean
    #curr_a = np.ones_like(mean) + np.random.random(size=(SIZE))

    n = curr_a.shape[0]
    initial_a = np.copy(curr_a)
    curr_a = norm_a(curr_a, mean)
    for evals in xrange(its):
        mv_samples = np.random.multivariate_normal(mean, cov, samples)
        print pearson(mean, curr_a)
        for i in xrange(n):
            mask = np.ones(curr_a.shape, dtype=bool)
            mask[i] = 0
            ai = curr_a[mask]
            yi = mv_samples.T[mask]
            yk = mv_samples.T[i]
            ak = dloss(ai, yi, yk)
            curr_a[i] = ak / samples
            #curr_a[i] = ak
        #curr_a = norm_a(curr_a, mean)
        print curr_a
        print np.mean(np.abs(initial_a - curr_a))
    return curr_a


def get_pearson(gp, test_data, samples=1000, its=10):
    feats = test_data[:, :-1]
    gold_labels = test_data[:, -1]
    mean, cov = gp.predict(feats, full_cov=True)
    mean = mean.flatten()
    prs_preds = prs_loss(mean, cov, samples=samples, its=its)
    r_mean = pearson(mean, gold_labels)
    r_loss = pearson(prs_preds.flatten(), gold_labels)
    return r_mean, r_loss


def query(gp, pool_data, random=False):
    if random:
        rand_i = np.random.randint(0, pool_data.shape[0])
        return pool_data[rand_i], rand_i
    else:
        means, vars = gp.predict(pool_data)
        max_var_i = np.argmax(vars.flatten())
        return pool_data[max_var_i], max_var_i


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
    data = read_data(FEATS_FILE, LABELS_FILE, size=SIZE)
    train_data = data[:SPLIT, :]
    test_data = data[SPLIT:, :]
    #train_data, scaler = normalize_train_data(train_data, hter=True)
    train_data, scaler = normalize_train_data(train_data, hter=False)
    test_data = normalize_test_data(test_data, scaler)

    gp = train_gp_model(train_data, kernel, warp, ard)
    print gp
    #print get_pearson(gp, test_data)
    #import ipdb; ipdb.set_trace()
    mae, rmse, ps, nlpd = get_metrics(gp, test_data)
    amaes = get_asym_metrics(gp, test_data)
    import pprint; pprint.pprint(amaes)
    import ipdb; ipdb.set_trace()
    print "MAE:\t\t%.4f" % mae
    #print "AMAE(3)(50):\t%.4f" % a_mae
    #print "AMAE(3)(75):\t%.4f" % a_mae_75
    print "RMSE:\t\t%.4f" % rmse
    print "Pearsons:\t%.4f\t%.4f" % ps
    print "NLPD:\t\t%.4f" % nlpd
    #save_cautious_curves(gp, test_data, 'test_curves_none_alt')
    #save_predictions(gp, test_data, 'test_preds_none_alt')
    #save_parameters(gp, '../saved_models/' + model)
    
