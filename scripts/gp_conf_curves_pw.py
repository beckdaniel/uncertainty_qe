import numpy as np
import scipy as sp
import GPy
import nltk
import sys
import sklearn.preprocessing as pp
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
import os
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
#from gpyk_wrapper.svm_wrapper import SVMWrapper
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy import interpolate
from scipy.stats import gaussian_kde
from sklearn.ensemble import BaggingRegressor as Bagging
import matplotlib.lines as mlines
#import Tango

NUM_SENTS = int(sys.argv[1])

DATA_DIR = '../data/eamt11_fr-en/shuf'
DATA_DIR = '../data/wmt14/shuf'
FEATS = np.loadtxt(os.path.join(DATA_DIR, 'features.shuf'))[:NUM_SENTS]
LABELS = np.loadtxt(os.path.join(DATA_DIR, 'time.shuf'), ndmin=2)[:NUM_SENTS]
#MASK = np.array([1,2,4,5,6,7,8,9,10,11,12,13,14])#,15,16])
#FEATS = FEATS[:,MASK]



#LABELS += TOL
if DATA_DIR == '../data/wmt14/shuf':
    LABELS = LABELS / 1000.

#for i in zip(LABELS[:100], FEATS[:,1:2][:100]):
#    print i, np.log((i[0] / i[1]))


#FEATS = pp.StandardScaler().fit_transform(FEATS)
#print FEATS

TOL = 1e-1
Y = LABELS
Y = LABELS / FEATS[:,1:2] # time per word
#Y = np.log(Y + TOL)

#Y = np.log(Y)
data = np.concatenate((FEATS, Y), axis=1)


#fig, axarr = plt.subplots(4,4)
MS = 2

SPLIT = (FEATS.shape[0] / 4) * 3

FOLDS = 5

def NLPD(preds, Y_test):
    means = preds[0]
    variances = preds[1]
    nlpd = (Y_test - means) ** 2
    nlpd /= variances
    nlpd += np.log(variances)
    return - np.mean(nlpd / 2)

def get_confidence_curve(preds, Y_test, random=False):
    outputs = preds[0]
    variances = preds[1]
    #print outputs.shape
    #print variances.shape
    #print Y_test.shape
    all_data = np.concatenate((outputs, Y_test, variances), axis=1)
    all_data = np.array(sorted(all_data, key=lambda x: x[2], reverse=False))
    if random:
        np.random.shuffle(all_data)
    #print all_data[:10]
    maes = []
    for i in range(1, len(all_data)):
        mae = MAE(all_data[0:i, 0], all_data[0:i, 1])
        maes.append(mae)
    return maes
    #plt.plot(range(50, len(all_data)), maes, c=color)
    #plt.show()
    

def train_model(model):
    #if isinstance(model, GPy.models.WarpedGP):
        #model['.*psi.*'].constrain_fixed(0)
        #model['.*tanh.d.*'].constrain_fixed(0)
        #model['.*lengthscale.*'].constrain_fixed(1)
    model.optimize_restarts(max_iters=100, messages=False, parallel=True, num_restarts=10, robust=True)
    print model
    #if isinstance(model, GPy.models.WarpedGP):
        #model['.*warp.*'].unconstrain()
        #model['.*lengthscale.*'].unconstrain()
        #model.optimize(max_iters=100, messages=False)
    print model
    print model['.*lengthscale.*']
    if isinstance(model, GPy.models.WarpedGP):
        preds = model.predict(X_test, median=True)
    else:
        preds = model.predict(X_test)
    #print preds[0]
    mae = MAE(preds[0], Y_test)
    rmse = np.sqrt(MSE(preds[0], Y_test))
    nlpd = NLPD(preds, Y_test)
    return preds, mae, rmse, nlpd
    #plot_confidence_curve(preds, Y_test, color)


gp_curves = []
gp_random_curves = []
wgp_curves = []
svm_curves = []
st_curves = []

gp_ll = []
wgp_ll = []
gp_maes = []
wgp_maes = []
gp_rmses = []
wgp_rmses = []
gp_nlpds = []
wgp_nlpds = []
print data.shape[0]
fold_size = data.shape[0] / FOLDS
for fold in range(FOLDS):
    data_train = np.concatenate((data[0:fold, :], data[fold+fold_size:, :])).copy()
    data_test = np.array(data[fold:fold+fold_size, :]).copy()

    X_train = data_train[:,:-1]
    X_test = data_test[:,:-1]
    Y_train = data_train[:,-1:]
    Y_test = data_test[:,-1:]

    #Y_train_pw = Y_train / X_train[:, 1:2]
    Y_train_pw = Y_train
    X_test_orig = X_test.copy()
    #print Y_train_pw

    print X_train.shape
    print X_test.shape

    feats_scaler = pp.StandardScaler().fit(X_train)
    X_train = feats_scaler.transform(X_train)
    X_test = feats_scaler.transform(X_test)


    #k = GPy.kern.RBF(X_train.shape[1], ARD=False)
    #k = GPy.kern.RBF(X_train.shape[1], ARD=True)
    #k = GPy.kern.Matern32(X_train.shape[1], ARD=False)
    k = GPy.kern.Matern32(X_train.shape[1], ARD=True)
    model = GPy.models.GPRegression(X_train, Y_train_pw, kernel=k)
    #model = GPy.models.WarpedGP(X_train, Y_train_pw, kernel=k)
    #model.predict_in_warped_space = True
    gp_preds, gp_mae, gp_rmse, gp_nlpd = train_model(model)
    new_gp_preds = gp_preds
    #new_gp_preds = []
    #new_gp_preds.append(gp_preds[0] * X_test_orig[:,1:2])
    #new_gp_preds.append(gp_preds[1])
    #new_gp_preds.append(np.sqrt(gp_preds[1]) * X_test_orig[:,1:2])
    gp_curves.append(get_confidence_curve(new_gp_preds, Y_test))
    gp_random_curves.append(get_confidence_curve(new_gp_preds, Y_test, random=True))
    gp_ll.append(model.log_likelihood())
    gp_maes.append(gp_mae)
    gp_rmses.append(gp_rmse)
    gp_nlpds.append(gp_nlpd)

    #k = GPy.kern.RBF(X_train.shape[1], ARD=True)
    #k = GPy.kern.RBF(X_train.shape[1], ARD=False)
    #model = GPy.models.GPRegression(X_train, Y_train_pw, kernel=k)
    #k = GPy.kern.Matern32(X_train.shape[1], ARD=True)
    #model = GPy.models.GPRegression(X_train, Y_train, kernel=k)
    k = GPy.kern.Matern32(X_train.shape[1], ARD=True)
    model = GPy.models.WarpedGP(X_train, Y_train_pw, kernel=k)
    model['warp_tanh.d'].constrain_fixed(0)
    model['warp.*'].constrain_fixed(1)
    #model.predict_in_warped_space = False
    wgp_preds, wgp_mae, wgp_rmse, wgp_nlpd = train_model(model)
    new_gp_preds = wgp_preds
    #new_gp_preds = []
    #new_gp_preds.append(wgp_preds[0] * X_test_orig[:,1:2])
    #new_gp_preds.append(gp_preds[1])
    #new_gp_preds.append(np.sqrt(wgp_preds[1]) * X_test_orig[:,1:2])
    wgp_curves.append(get_confidence_curve(new_gp_preds, Y_test))
    wgp_ll.append(model.log_likelihood())
    wgp_maes.append(wgp_mae)
    wgp_rmses.append(wgp_rmse)
    wgp_nlpds.append(wgp_nlpd)

    # k = GPy.kern.Matern32(X_train.shape[1], ARD=True)
    # ll = GPy.likelihoods.StudentT()
    # inf = GPy.inference.latent_function_inference.Laplace()
    # model = GPy.core.GP(X_train, Y_train, kernel=k, likelihood=ll, inference_method=inf)
    # model.predict_in_warped_space = True
    # st_preds, st_mae = train_model(model)
    # st_curves.append(get_confidence_curve(st_preds, Y_test))



    tuned = [{'C': np.logspace(-2,3,4),
              'gamma': np.logspace(-3,2,4),
              'epsilon': np.logspace(-2,3,4)}]
    #svm = Bagging(GridSearchCV(SVR(), tuned), n_jobs=4, n_estimators=20)
    #svm.fit(X_train, Y_train_pw.flatten())
    #svm_preds = [e.predict(X_test) for e in svm.estimators_]
    #svm_preds = [e.predict(X_test) * X_test_orig[:, 1] for e in svm.estimators_]
    #svm_means = np.mean(svm_preds, axis=0)[:, None]
    #svm_vars = (np.std(svm_preds, axis=0) ** 1)[:, None]
    #svm_means = np.mean(svm_preds, axis=0)[:, None] * X_test_orig[:, 1:2]
    #svm_curves.append(get_confidence_curve((svm_means, svm_vars), Y_test))
    #wgp_curves = svm_curves
    #import ipdb;ipdb.set_trace()

    #break


print 'LL:'
print np.mean(gp_ll)
print np.mean(wgp_ll)
print 'MAES:'
print np.mean(gp_maes)
print np.mean(wgp_maes)
print 'RMSES:'
print np.mean(gp_rmses)
print np.mean(wgp_rmses)
print 'NLPDS:'
print np.mean(gp_nlpds)
print np.mean(wgp_nlpds)


gp_mean = np.mean(gp_curves, axis=0)
gp_std = np.std(gp_curves, axis=0)
wgp_mean = np.mean(wgp_curves, axis=0)
wgp_std = np.std(wgp_curves, axis=0)
#svm_mean = np.mean(svm_curves, axis=0)
#svm_std = np.std(svm_curves, axis=0)
x_plot = np.array(range(1, data.shape[0] / FOLDS))

gp_random_mean = np.mean(gp_random_curves, axis=0)
gp_random_std = np.std(gp_random_curves, axis=0)


from GPy.plotting.matplot_dep.base_plots import gpplot


fig, axarr = plt.subplots(1,1)
gpplot(x_plot, gp_mean, (gp_mean - (2*gp_std)),
       (gp_mean + (2*gp_std)), ax=axarr)
gpplot(x_plot, wgp_mean, (wgp_mean - (2*wgp_std)),
       (wgp_mean + (2*wgp_std)), ax=axarr,
       edgecol='DarkGreen',
       fillcol='LightGreen')
gpplot(x_plot, gp_random_mean, (gp_random_mean - (2*gp_random_std)),
       (gp_random_mean + (2*gp_random_std)), ax=axarr,
       edgecol='k',
       fillcol='LightGray')
#gpplot(x_plot, svm_mean, (svm_mean - (2*svm_std)),
#       (svm_mean + (2*svm_std)), ax=axarr,
#       edgecol='DarkGreen',
#       fillcol='LightGreen')

#print gp_curves
#print wgp_curves
#plt.plot(range(1, data.shape[0] / FOLDS), np.mean(gp_curves, axis=0), c='b')
#plt.plot(range(1, data.shape[0] / FOLDS), np.mean(wgp_curves, axis=0), c='g')
#plt.plot(range(1, data.shape[0] / FOLDS), np.mean(st_curves, axis=0), c='k')

#plt.legend(['RBF', 'Matern32'])
#plt.legend(['GP Mat32 ARD', 'WarpGP Mat32 ARD'])
#plt.legend(['GP Mat32 ARD', 'WarpGP Mat32 ARD', 'GP Mat32 ARD (random)'])
plt.legend(handles=[mlines.Line2D([],[],color='DarkBlue', label='WGP Mat32 ISO', lw=3),
                    mlines.Line2D([],[],color='DarkGreen', label='WGP Mat32 ARD', lw=3),
                    mlines.Line2D([],[],color='k', label='WGP Mat32 ISO (random)', lw=3),
                    ])
    

#model = GPy.models.WarpedGP(X_train, Y_train, kernel=k)
#model.predict_in_warped_space = True



#k = GPy.kern.RBF(X_train.shape[1], ARD=True)
#model = GPy.models.GPRegression(X_train, Y_train, kernel=k)
#plot_model(model, 'b')

#k = GPy.kern.RBF(X_train.shape[1], ARD=True)
#model = GPy.models.WarpedGP(X_train, Y_train, kernel=k)
#model.predict_in_warped_space = True
#plot_model(model, 'g')




#k = GPy.kern.Matern52(X_train.shape[1], ARD=True)
#model = GPy.models.GPRegression(X_train, Y_train, kernel=k)
#plot_model(model, 'k')
plt.show()
