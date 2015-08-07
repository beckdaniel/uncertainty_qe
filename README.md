# uncertainty_qe
Experiments to address uncertainty in Quality Estimation

# Requirements

- Python 2.7
- Numpy
- Matplotlib
- scikit-learn (for SVMs)
- GPy (warped_gp_fixes branch in beckdaniel's fork)

# Data

3 datasets:

- WMT14 English-Spanish
- EAMT11 English-Spanish
- EAMT11 French-English

Features are the 17 baseline Quest features. Response variables are post-editing time per word (check Graham(2015), ACL).

# Experiments sketch

For each dataset, we perform experiments using the following ML models:

- SVM + Bagging
- GP RBF Kernel Isotropic
- GP RBF Kernel ARD
- GP Matern32 Kernel ARD
- Warped GP Matern32 Kernel ARD

Intrinsic evaluation is made using the following metrics:

- NLPD (only available for GPs)
- MAE
- MSE
- Pearson's correlation measure (check Graham(2015), ACL)

Extrinsic evaluation is made using the following tasks:

- Reject option setting
- Active learning setting
- ?

Everything is done via cross-validation (5 folds as default but this can be changed).

# Reject options

Main idea: ignore predicitons with high uncertainty (variance). We plot curves measuring
intrinsic metrics (Pearson's?) according to the top N% most confident predictions. Ideal curves should
be monotonic.

# Active learning

The setting is similar to Beck et al. (2013, ACL). Start with a small (default: 50) set, measure
error (Pearson's?) on test set and use active learning to incrementally increase the size of the
training set. Error should eventually reach a plateau with very few sentences. An oracle setting
is also available.


