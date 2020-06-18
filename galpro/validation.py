import numpy as np
import h5py
import os
from galpro.plot import Plot


class Validation:

    def __init__(self, target_features, path):
        self.target_features = target_features
        self.path = path
        self.no_features = len(target_features)
        self.no_points = 100

        # Initialise classes
        self.plot = Plot(target_features=self.target_features, path=self.path)

    def validate(self, y_test, pdfs=None, save_validation=False, make_plots=False):

        no_samples = y_test.shape[0]

        # Check if posterior() has been run
        folder = '/posteriors/'
        if pdfs is None:
            # If not then try to load any saved posteriors
            pdfs = []
            if os.path.isdir(self.path + folder):
                for sample in np.arange(no_samples):
                    pdf = h5py.File(self.path + folder + str(sample) + ".h5", "r")
                    pdfs.append(pdf['data'][:])
            else:
                print('No posteriors have been found. Please run Model.posterior()'
                      'to generate posteriors and then run validate() or pass in the pdfs.')
                exit()

        # Probabilistic calibration
        pits = np.empty((no_samples, self.no_features))
        marginal_outliers = np.empty(self.no_features)

        for feature in np.arange(self.no_features):
            for sample in np.arange(no_samples):
                pdf = np.array(pdfs[sample])
                pits[sample, feature] = np.sum(pdf[:, feature] <= y_test[sample, feature]) / pdf.shape[0]

        # Marginal calibration
        marginal_calibration = np.empty((self.no_points, self.no_features))

        for feature in np.arange(self.no_features):
            count = 0
            min_, max_ = [np.int(np.min(y_test[:, feature])), np.int(np.max(y_test[:, feature]))]

            for point in np.linspace(min_, max_, self.no_points):
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
        template_pred, template_true = self._create_templates()

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
        kendall_calibration = np.empty(self.no_points)
        count = 0

        for point in np.linspace(0, 1, self.no_points):
            sum_ = np.zeros(no_samples)
            for sample in np.arange(no_samples):
                sum_[sample] = np.sum(pred_cdf_full[sample] <= point) / len(pred_cdf_full[sample])

            kendall_func_point = np.sum(sum_) / no_samples
            true_cdf_point = np.sum(true_cdf_full <= point) / no_samples
            kendall_calibration[count] = kendall_func_point - true_cdf_point
            count += 1

        folder = '/validation/'
        if save_validation:
            if os.path.isdir(self.path + folder):
                print('Previously saved validation has been overwritten')
            else:
                os.mkdir(self.path + folder)
            np.save(self.path + folder + 'pits.npy', pits)
            np.save(self.path + folder + 'marginal_calibration.npy', marginal_calibration)
            np.save(self.path + folder + 'coppits.npy', coppits)
            np.save(self.path + folder + 'kendall_calibration.npy', kendall_calibration)

        if make_plots:
            self.plot.plot_pit(pit=pits)
            self.plot.plot_coppit(coppit=coppits)
            self.plot.plot_marginal_calibration(marginal_calibration=marginal_calibration, y_test=y_test)
            self.plot.plot_kendall_calibration(kendall_calibration=kendall_calibration)

        return pits, marginal_calibration, coppits, kendall_calibration

    def _create_templates(self):

        template = []
        for feature in np.arange(self.no_features):

            if feature != (self.no_features - 1):
                template.append('(pdf[:,' + str(feature) + '] < pdf[pred,' + str(feature) + ']) & ')
            else:
                template.append('(pdf[:,' + str(feature) + '] < pdf[pred,' + str(feature) + '])')

        template_pred = ''.join(template)
        template_true = template_pred.replace('pdf[pred', 'y_test[sample')
        template_true = template_true.replace('<', '<=')

        return template_pred, template_true
