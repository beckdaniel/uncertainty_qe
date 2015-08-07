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

NUM_SENTS = int(sys.argv[1])

DATA_DIR = '../data/eamt11_fr-en/shuf'
DATA_DIR = '../data/wmt14/shuf'
FEATS = np.loadtxt(os.path.join(DATA_DIR, 'features.shuf'))[:NUM_SENTS]
LABELS = np.loadtxt(os.path.join(DATA_DIR, 'time.shuf'), ndmin=2)[:NUM_SENTS]



def plot_confidence_curve(preds, Y_test):
    outputs = preds[0]
    variances = preds[1]
    all_data = np.concatenate((outputs, Y_test, variances), axis=1)
    all_data = sorted(all_data, key=lambda x: x[2])
    #print all_data[:10]
    maes = []
    for i in range(10, len(all_data)):
        mae = MAE(outputs[0:i], Y_test[0:i])
        maes.append(mae)
    plt.plot(range(10, len(all_data)), maes)
    plt.show()
    

def NLPD(preds, Y_test):
    means = preds[0]
    variances = preds[1]
    nlpd = (Y_test - means) ** 2
    nlpd /= variances
    nlpd += np.log(variances)
    return - np.mean(nlpd / 2)
    

#LABELS += TOL
if DATA_DIR == '../data/wmt14/shuf':
    LABELS = LABELS / 1000.

#for i in zip(LABELS[:100], FEATS[:,1:2][:100]):
#    print i, np.log((i[0] / i[1]))



#print FEATS

TOL = 1e-1
Y = LABELS
Y = LABELS / FEATS[:,1:2] # time per word
#Y = np.log(Y + TOL)

#Y = np.log(Y)
FEATS = pp.StandardScaler().fit_transform(FEATS)
data = np.concatenate((FEATS, Y), axis=1)


######################
def afilter(FEATS, Y):
    to_filter = np.where(Y > -10)[0]
    new_feats = FEATS[to_filter,:]
    new_y = Y[to_filter,:]
    return new_feats, new_y

#FEATS, Y = afilter(FEATS, Y)
######################

fig, axarr = plt.subplots(4,4)
MS = 2

SPLIT = FEATS.shape[0] - (FEATS.shape[0] / 4)

X_train = FEATS[:SPLIT,:]
X_test = FEATS[SPLIT:,:]
Y_train = Y[:SPLIT,:]
Y_test = Y[SPLIT:,:]

k = GPy.kern.Matern32(17, ARD=True)
#k = GPy.kern.RBF(17, ARD=True)
#k = GPy.kern.RBF(17, ARD=False)
#model = GPy.models.GPRegression(X_train, Y_train, kernel=k)
model = GPy.models.WarpedGP(X_train, Y_train, kernel=k)
model.predict_in_warped_space = True

#model.optimize(max_iters=100, messages=True)
model.optimize_restarts(max_iters=100, num_restarts=10, parallel=True, robust=True)
print model
print model['.*lengthscale.*']
preds = model.predict(X_test)
#print preds[0]
print 'MAE'
print MAE(preds[0], Y_test)
print 'RMSE'
print np.sqrt(MSE(preds[0], Y_test))
print 'NLPD'
print NLPD(preds, Y_test)
#plot_confidence_curve(preds, Y_test)
#sys.exit(0)

for i in range(16):
    x_plot = i / 4
    y_plot = i % 4

    fixed_inputs = []
    for j in range(17):
        if i != j:
            #fixed_inputs.append((j,0))
            fixed_inputs.append((j, np.mean(FEATS[:,j])))
    model.plot(ax=axarr[x_plot,y_plot], fixed_inputs=fixed_inputs)
    
    #print i+1, model.log_likelihood()
    #print model


axarr[0,0].set_title('sentence length (src)')
axarr[0,1].set_title('sentence length (tgt)')
axarr[0,2].set_title('avg token length (src)')
axarr[0,3].set_title('LM prob (src)')
axarr[1,0].set_title('LM prob (tgt)')
axarr[1,1].set_title('type/token ratio (tgt)')
axarr[1,2].set_title('avg #translations per word (src, p > 0.2)')
axarr[1,3].set_title('avg #translations per word (src, p > 0.01)')
axarr[2,0].set_title('1-gram percentage in q1 (src)')
axarr[2,1].set_title('1-gram percentage in q4 (src)')
axarr[2,2].set_title('2-gram percentage in q1 (src)')
axarr[2,3].set_title('2-gram percentage in q4 (src)')
axarr[3,0].set_title('3-gram percentage in q1 (src)')
axarr[3,1].set_title('3-gram percentage in q4 (src)')
axarr[3,2].set_title('1-gram overall percentage (src)')
axarr[3,3].set_title('number of punctuation marks (src)')


fig.subplots_adjust(wspace=0.15, hspace=0.2, left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.show()
sys.exit(0)

