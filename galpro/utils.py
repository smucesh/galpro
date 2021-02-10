from sys import exit
import os
import h5py
import numpy as np
from scipy.stats import entropy, kstest, median_abs_deviation, cramervonmises


def create_directories(model_name):

    # Define directories
    path = os.getcwd() + '/galpro/' + str(model_name) + '/'
    point_estimate_folder = 'point_estimates/'
    posterior_folder = 'posteriors/'
    validation_folder = 'validation/'
    plot_folder = 'plots/'

    # Create root directory
    if not os.path.isdir(os.getcwd() + '/galpro/'):
        os.mkdir(os.getcwd() + '/galpro/')

    # Create model directory
    os.mkdir(path)
    os.mkdir(path + point_estimate_folder)
    os.mkdir(path + posterior_folder)
    os.mkdir(path + validation_folder)
    os.mkdir(path + point_estimate_folder + plot_folder)
    os.mkdir(path + posterior_folder + plot_folder)
    os.mkdir(path + validation_folder + plot_folder)


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

    point_estimate_folder = 'point_estimates/'
    if os.path.isfile(path + point_estimate_folder + 'point_estimates.h5'):
        with h5py.File(path + point_estimate_folder + "point_estimates.h5", 'r') as f:
            y_pred = f['point_estimates'][:]
        print('Previously saved point estimates have been loaded.')
    else:
        print('Point estimates have not been found. Run point_estimates().')
        exit()

    return y_pred


def load_posteriors(path):
    """Loads saved posteriors."""

    posterior_folder = 'posteriors/'
    if os.path.isfile(path + posterior_folder + 'posteriors.h5'):
        posteriors = h5py.File(path + posterior_folder + "posteriors.h5", "r")
        print('Previously saved posteriors have been loaded.')
    else:
        print('No posteriors have been found. Run posterior() to generate posteriors.')
        exit()

    return posteriors


def load_validation(path):
    """Loads different calibrations"""

    validation_folder = 'validation/'
    if os.path.isfile(path + validation_folder + 'validation.h5'):
        validation = h5py.File(path + validation_folder + "validation.h5", "r")
        print('Previously saved validation has been loaded.')
    else:
        print('No validation has been found. Run validate().')
        exit()

    return validation


def get_pred_metrics(y_test, y_pred, no_features):
    """Calculates performance metrics for point predictions."""

    metrics = np.empty(no_features)
    for feature in np.arange(no_features):
        nmad = median_abs_deviation(y_pred[:, feature]-y_test[:, feature], scale=1/1.4826)
        metrics[feature] = nmad

    return metrics


def get_pdf_metrics(pits, no_samples, no_features, no_bins, coppits=None):
    """Calculates performance metrics for PDFs."""

    pit_outliers = np.empty(no_features)
    pit_kld = np.empty(no_features)
    pit_kst = np.empty(no_features)
    pit_cvm = np.empty(no_features)

    for feature in np.arange(no_features):
        pit_pdf, pit_bins = np.histogram(pits[:, feature], density=True, bins=no_bins)
        uniform_pdf = np.full(no_bins, 1.0/no_bins)
        pit_kld[feature] = entropy(pit_pdf, uniform_pdf)
        pit_kst[feature] = kstest(pits[:, feature], 'uniform')[0]
        pit_cvm[feature] = cramervonmises(pits[:, feature], 'uniform').statistic
        no_outliers = np.count_nonzero(pits[:, feature] == 0) + np.count_nonzero(pits[:, feature] == 1)
        pit_outliers[feature] = (no_outliers / no_samples) * 100

    if coppits is not None:
        coppit_pdf, coppit_bins = np.histogram(coppits, density=True, bins=no_bins)
        uniform_pdf = np.full(no_bins, 1.0 / no_bins)
        coppit_kld = entropy(coppit_pdf, uniform_pdf)
        coppit_kst = kstest(coppits, 'uniform')[0]
        coppit_cvm = cramervonmises(coppits, 'uniform').statistic
        no_outliers = len(set(np.where((pits == 0) | (pits == 1))[0]))
        coppit_outliers = (no_outliers/no_samples) * 100
        return coppit_outliers, coppit_kld, coppit_kst, coppit_cvm

    return pit_outliers, pit_kld, pit_kst, pit_cvm


def get_quantiles(posteriors, no_samples, no_features):
    """Calculate the 16th, 50th and 84th quantiles."""

    quantiles = np.empty((no_features, no_samples, 3))

    for feature in np.arange(no_features):
        for sample in np.arange(no_samples):
            posterior = posteriors[str(sample)][:]
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
