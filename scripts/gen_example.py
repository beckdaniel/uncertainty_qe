import numpy as np
import GPy
import sklearn.preprocessing as pp
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from scipy.stats import pearsonr as pearson
import json
import util
import sys

TRAIN = sys.argv[1]
TEST = sys.argv[2]
PARAMS = sys.argv[3]

train_data = np.loadtxt(TRAIN)#[:100]
test_data = np.loadtxt(TEST)#[:2]

k = GPy.kern.Matern52(17, ARD=True)
#gp = GPy.models.GPRegression(train_data[:, :-1], train_data[:, -1:], kernel=k)
w = GPy.util.warping_functions.TanhFunction(n_terms=3)
gp = GPy.models.WarpedGP(train_data[:, :-1], train_data[:, -1:], kernel=k, warping_function=w)

util.load_parameters(gp, PARAMS)
print gp
print gp['.*lengthscale.*']

preds = gp.predict(test_data[:, :-1], median=True)
#preds = gp.predict(test_data[:, :-1])
nlpd = gp.log_predictive_density(test_data[:, :-1], test_data[:, -1:]).flatten()
Q = tuple(range(1, 100))
quantiles = np.array(gp.predict_quantiles(test_data[:, :-1], quantiles=Q)).reshape((99, test_data.shape[0])).T
#print preds
means = preds[0].flatten()
variances = preds[1].flatten()
gold = test_data[:, -1]
result = []
for tup in zip(gold-means, means, nlpd, gold, quantiles):
    result.append(np.concatenate(([tup[0]], [tup[1]], [tup[2]], [tup[3]], tup[4])))
result = np.array(result)
#print result[0]
np.savetxt('examples/fr-en_tanh3_true_result', result, fmt='%.5f')
#np.savetxt('examples/fr-en_none_true_result', result, fmt='%.5f')
