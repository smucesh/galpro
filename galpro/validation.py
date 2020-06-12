import numpy as np
import h5py
import os


def validate(y_test, pdfs, path, make_plots=False, save_validation=False):

    no_samples, no_features = [y_test.shape[0], y_test.shape[1]]
    no_points = 100

    # Check if posterior() has been run
    if pdfs is None:
        # If not then try to load any saved posteriors
        pdfs = []
        if os.path.isdir('model/posterior'):
            for sample in np.arange(no_samples):
                pdf = h5py.File(path + 'posterior/' + str(sample) + ".h5", "r")
                pdfs.append(pdf['data'][:])
        else:
            print('No posteriors have been found. Please run Model.posterior()'
                  'to generate posteriors and then run validate().')

        # Probabilistic calibration
        pits = np.empty((no_samples, no_features))
        marginal_outliers = np.empty(no_features)

        for feature in np.arange(no_features):
            for sample in np.arange(no_samples):
                pdf = np.array(pdfs[sample])
                pits[sample] = np.sum(pdf[:, feature] <= y_test[sample, feature]) / pdf.shape[0]

            no_outliers = np.count_nonzero(pits[:, feature] == 0) + np.count_nonzero(pits[:, feature] == 1)
            marginal_outliers[feature] = (no_outliers / no_samples) * 100

        # Marginal calibration
        marginal_calibration = np.empty((no_points, no_features))

        for feature in np.arange(no_features):
            count = 0
            min_, max_ = [np.int(np.min(y_test[:, feature])), np.int(np.max(y_test[:, feature]))]

            for point in np.linspace(min_, max_, no_points):
                sum_ = np.zeros(no_samples)
                for sample in np.arange(no_samples):
                    pdf = np.array(pdfs[sample])
                    sum_[sample] = np.sum(pdf[:, feature] <= point) / pdf.shape[0]

                pred_cdf_marg_point = np.sum(sum_) / no_samples
                true_cdf_marg_point = np.sum(y_test[:, feature] <= point) / no_samples
                marginal_calibration[count, feature] = pred_cdf_marg_point - true_cdf_marg_point
                count += 1

        # Probabilistic copula calibration
        # Creating a list of list containing pred_cdf of each point in predictions
        pred_cdf_full = [[] for i in np.arange(no_samples)]
        true_cdf_full = []
        coppits = np.empty(no_samples)
        template_pred, template_true = _create_templates(no_features)

        for sample in np.arange(no_samples):
            pdf = np.array(pdfs[sample])
            no_preds = pdf.shape[0]

            for pred in np.arange(no_preds):
                # For point at edges, if <= used, then point counts and cdf is never 0.
                # If <= is used, a large number of point will have near 0 probability, as a result, there will
                # be a peak at 0.
                # -1 insures, the point in consideration does not count when determining cdf.
                pred_cdf_full[sample].append(np.sum(eval(template_pred)) / (no_preds - 1))

            true_cdf_full.append(np.sum(eval(template_true)) / no_preds)
            coppits[sample] = np.sum(pred_cdf_full[sample] <= true_cdf_full[sample]) / no_preds

        # Kendall calibration
        kendall_calibration = np.empty(no_points)
        count = 0

        for point in np.linspace(0, 1, no_points):
            sum_ = np.zeros(no_samples)
            for sample in np.arange(no_samples):
                sum_[sample] = np.sum(pred_cdf_full[sample] <= point) / len(pred_cdf_full[sample])

            kendall_func_point = np.sum(sum_) / no_samples
            true_cdf_point = np.sum(true_cdf_full <= point) / no_samples
            kendall_calibration[count] = kendall_func_point - true_cdf_point
            count += 1

        return pits, marginal_calibration, coppits, kendall_calibration


def _create_templates(no_features):

    template = []
    for feature in np.arange(no_features):

        if feature != (no_features - 1):
            template.append('(pdf[:,' + str(feature) + '] < pdf[pred,' + str(feature) + ']) & ')
        else:
            template.append('(pdf[:,' + str(feature) + '] < pdf[pred,' + str(feature) + '])')

    template_pred = ''.join(template)
    template_true = template_pred.replace('pdf[pred', 'y_test[sample')
    template_true = template_true.replace('<', '<=')

    return template_pred, template_true
