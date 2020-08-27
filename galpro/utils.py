import os
import h5py
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import entropy, uniform
from skgof import ks_test, cvm_test


def convert_1d_arrays(*arrays):
    arrays = list(arrays)
    for i in np.arange(len(arrays)):
        array = arrays[i]
        if array is not None:
            arrays[i] = arrays[i].reshape(-1, 1)

    return arrays


def load_point_estimates(path):

    y_pred = []
    point_estimate_folder = 'point_estimates/'
    if os.path.isfile(path + point_estimate_folder + 'point_estimates.npy'):
        y_pred = np.load(path + point_estimate_folder + 'point_estimates.npy')
        print('Previously saved point estimates have been loaded.')
    else:
        print('Point estimates have not been found. Please run point_estimates().')
        exit()

    return y_pred


def load_posteriors(path):

    pdfs = []
    posterior_folder = 'posteriors/'
    no_samples = len(os.listdir(path + posterior_folder)) - 1

    if no_samples != 0:
        for sample in np.arange(no_samples):
            pdf = h5py.File(path + posterior_folder + str(sample) + ".h5", "r")
            pdfs.append(pdf['data'][:])
        print('Previously saved posteriors have been loaded.')
    else:
        print('No posteriors have been found. Run posterior() to generate posteriors.')
        exit()

    return pdfs


def get_pred_metrics(y_test, y_pred):

    no_features = y_pred.shape[1]
    rmses = []
    for feature in np.arange(no_features):
        rmse = mean_squared_error(y_true=y_test[:, feature], y_pred=y_pred[:, feature], squared=False)
        rmses.append(np.around(rmse, 3))

    return rmses


def get_pdf_metrics(data, no_features):

    no_bins = 100
    no_samples = data.shape[0]
    outliers = np.empty(no_features)
    kld = np.empty(no_features)
    kst = np.empty(no_features)
    cvm = np.empty(no_features)

    if no_features > 1:
        template = 'data[:, feature]'
    else:
        template = 'data'

    for feature in np.arange(no_features):
        pit_pdf, pit_bins = np.histogram(eval(template), density=True, bins=no_bins)
        uniform_pdf = np.full(no_bins, 1.0/no_bins)
        kld[feature] = entropy(pit_pdf, uniform_pdf)
        kst[feature] = ks_test(eval(template), uniform(0, 1))[0]
        cvm[feature] = cvm_test(eval(template), uniform(0, 1))[0]
        no_outliers = np.count_nonzero(eval(template) == 0) + np.count_nonzero(eval(template) == 1)
        outliers[feature] = (no_outliers/no_samples) * 100

    return outliers, kld, kst, cvm


def get_quantiles(pdfs):

    no_samples = len(pdfs)
    no_features = len(pdfs[0][0])
    quantiles = np.empty((no_features, no_samples, 3))

    for feature in np.arange(no_features):
        for sample in np.arange(no_samples):
            pdf = np.array(pdfs[sample])
            quantiles[feature, sample] = np.percentile(a=pdf[:, feature], q=[16, 50, 84])

    return quantiles
