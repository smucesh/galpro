import os
import h5py
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import entropy, kstest, median_abs_deviation


def convert_1d_arrays(*arrays):
    """Convert given 1d arrays from shape (n,) to (n, 1) for compatibility with code."""

    arrays = list(arrays)
    for i in np.arange(len(arrays)):
        array = arrays[i]
        if array is not None:
            arrays[i] = arrays[i].reshape(-1, 1)

    return arrays


def load_point_estimates(path):
    """Loads saved point estimates."""

    y_pred = []
    point_estimate_folder = 'point_estimates/'
    if os.path.isfile(path + point_estimate_folder + 'point_estimates.npy'):
        y_pred = np.load(path + point_estimate_folder + 'point_estimates.npy')
        print('Previously saved point estimates have been loaded.')
    else:
        print('Point estimates have not been found. Run point_estimates().')
        exit()

    return y_pred


def load_posteriors(path):
    """Loads saved posteriors."""

    posteriors = []
    posterior_folder = 'posteriors/'
    no_samples = len(os.listdir(path + posterior_folder)) - 1

    if no_samples != 0:
        for sample in np.arange(no_samples):
            posterior = h5py.File(path + posterior_folder + str(sample) + ".h5", "r")
            posteriors.append(posterior['data'][:])
        print('Previously saved posteriors have been loaded.')
    else:
        print('No posteriors have been found. Run posterior() to generate posteriors.')
        exit()

    return posteriors


def load_calibration(path, calibration_mode):
    """Loads different calibrations"""

    validation_folder = 'validation/'
    calibration = None
    if os.path.isfile(path + validation_folder + calibration_mode + '.npy'):
        calibration = np.load(path + validation_folder + calibration_mode + '.npy')
        print('Previously saved ' + calibration_mode + ' has been loaded.')
    else:
        print(calibration_mode + ' has not been found. Run validate().')
        exit()

    return calibration


def get_pred_metrics(y_test, y_pred):
    """Calculates performance metrics for point predictions."""

    '''
    no_features = y_pred.shape[1]
    rmses = []
    for feature in np.arange(no_features):
        rmse = mean_squared_error(y_true=y_test[:, feature], y_pred=y_pred[:, feature], squared=False)
        rmses.append(np.around(rmse, 3))
    '''

    no_features = y_pred.shape[1]
    mads = []
    for feature in np.arange(no_features):
        mad = median_abs_deviation(y_pred[:, feature]-y_test[:, feature], scale=1/1.4826)
        mads.append(mad)

    return mads


def get_pdf_metrics(data, no_features, path=None):
    """Calculates performance metrics for PDFs."""

    no_bins = 100
    no_samples = data.shape[0]
    outliers = np.empty(no_features)
    kld = np.empty(no_features)
    kst = np.empty(no_features)
    #cvm = np.empty(no_features)

    if no_features > 1:
        template = 'data[:, feature]'
    else:
        template = 'data'

    for feature in np.arange(no_features):
        pit_pdf, pit_bins = np.histogram(eval(template), density=True, bins=no_bins)
        uniform_pdf = np.full(no_bins, 1.0/no_bins)
        kld[feature] = entropy(pit_pdf, uniform_pdf)
        kst[feature] = kstest(eval(template), 'uniform')[0]
        if no_features > 1:
            no_outliers = np.count_nonzero(eval(template) == 0) + np.count_nonzero(eval(template) == 1)
        else:
            pits = load_calibration(path=path, calibration_mode='pits')
            no_outliers = len(set(np.where((pits == 0) | (pits == 1))[0]))
        outliers[feature] = (no_outliers/no_samples) * 100

    return outliers, kld, kst


def get_quantiles(posteriors):
    """Calculate the 16th, 50th and 84th quantiles."""

    no_samples = len(posteriors)
    no_features = len(posteriors[0][0])
    quantiles = np.empty((no_features, no_samples, 3))

    for feature in np.arange(no_features):
        for sample in np.arange(no_samples):
            posterior = np.array(posteriors[sample])
            quantiles[feature, sample] = np.percentile(a=posterior[:, feature], q=[16, 50, 84])

    return quantiles


def create_templates(no_features):
    """Creates templates to perform multivariate calibration"""

    template = []
    template_same = []
    for feature in np.arange(no_features):

        if feature != (no_features - 1):
            template.append('(posterior[:,' + str(feature) + '] < posterior[pred,' + str(feature) + ']) & ')
            template_same.append('(posterior[:,' + str(feature) + '] == posterior[pred,' + str(feature) + ']) & ')
        else:
            template.append('(posterior[:,' + str(feature) + '] < posterior[pred,' + str(feature) + '])')
            template_same.append('(posterior[:,' + str(feature) + '] == posterior[pred,' + str(feature) + '])')

    template_pred = ''.join(template)
    template_same = ''.join(template_same)
    template_true = template_pred.replace('posterior[pred', 'self.y_test[sample')
    template_true = template_true.replace('<', '<=')

    return template_pred, template_true, template_same
