import numpy as np

from galpro.plot import Plot
from galpro.utils import load_posteriors


class Validation:

    def __init__(self, y_test, target_features, path):

        # Initialise arguments
        self.y_test = y_test
        self.path = path
        self.target_features = target_features
        self.pdfs = []
        self.no_features = len(target_features)
        self.no_points = 100
        self.posterior_folder = 'posteriors/'
        self.validation_folder = 'validation/'

        if self.y_test is not None:
            self.no_samples = self.y_test.shape[0]

        # Initialise classes
        self.plot = Plot(y_test=self.y_test, target_features=self.target_features, path=self.path)

    def validate(self):

        if self.y_test is None:
            print('Pass in y_test to run validation.')
            exit()

        # Load posteriors
        self.pdfs = load_posteriors(path=self.path)

        # Run validation
        self.probabilistic_calibration()
        self.marginal_calibration()
        self.plot.plot_pit()
        self.plot.plot_marginal_calibration()

        if self.no_features > 1:
            pred_cdf_full, true_cdf_full = self.probabilistic_copula_calibration()
            self.kendall_calibration(pred_cdf_full, true_cdf_full)
            self.plot.plot_coppit()
            self.plot.plot_kendall_calibration()

        print('Saved validation. Any previously saved validation has been overwritten.')

    def probabilistic_calibration(self):

        pits = np.empty((self.no_samples, self.no_features))

        for feature in np.arange(self.no_features):
            for sample in np.arange(self.no_samples):
                pdf = np.array(self.pdfs[sample])
                pits[sample, feature] = np.sum(pdf[:, feature] <= self.y_test[sample, feature]) / pdf.shape[0]

        print('Completed probabilistic calibration.')
        np.save(self.path + self.validation_folder + 'pits.npy', pits)

    def marginal_calibration(self):

        marginal_calibration = np.empty((self.no_points, self.no_features))

        for feature in np.arange(self.no_features):
            count = 0
            min_, max_ = [np.floor(np.min(self.y_test[:, feature])), np.ceil(np.max(self.y_test[:, feature]))]

            for point in np.linspace(min_, max_, self.no_points):
                sum_ = np.zeros(self.no_samples)
                for sample in np.arange(self.no_samples):
                    pdf = np.array(self.pdfs[sample])
                    sum_[sample] = np.sum(pdf[:, feature] <= point) / pdf.shape[0]

                pred_cdf_marg_point = np.sum(sum_) / self.no_samples
                true_cdf_marg_point = np.sum(self.y_test[:, feature] <= point) / self.no_samples
                marginal_calibration[count, feature] = pred_cdf_marg_point - true_cdf_marg_point
                count += 1

        np.save(self.path + self.validation_folder + 'marginal_calibration.npy', marginal_calibration)
        print('Completed marginal calibration.')

    def probabilistic_copula_calibration(self):

        # Creating a list of list containing pred_cdf of each point in predictions
        pred_cdf_full = [[] for i in np.arange(self.no_samples)]
        true_cdf_full = []
        coppits = np.empty(self.no_samples)
        template_pred, template_true, template_same = self._create_templates()

        for sample in np.arange(self.no_samples):
            pdf = np.array(self.pdfs[sample])
            no_preds = pdf.shape[0]

            for pred in np.arange(no_preds):
                # For point at edges, if <= used, then point counts and cdf is never 0.
                # If <= is used, a large number of point will have near 0 probability, as a result, there will
                # be a peak at 0.
                # -1 insures, the point in consideration does not count when determining cdf.
                same_preds = np.sum(eval(template_same))
                pred_cdf_full[sample].append(np.sum(eval(template_pred)) / (no_preds - same_preds))

            true_cdf_full.append(np.sum(eval(template_true)) / no_preds)
            coppits[sample] = np.sum(pred_cdf_full[sample] <= true_cdf_full[sample]) / no_preds

        np.save(self.path + self.validation_folder + 'coppits.npy', coppits)
        print('Completed probabilistic copula calibration')

        return pred_cdf_full, true_cdf_full

    def kendall_calibration(self, pred_cdf_full, true_cdf_full):

        kendall_calibration = np.empty(self.no_points)
        count = 0

        for point in np.linspace(0, 1, self.no_points):
            sum_ = np.zeros(self.no_samples)
            for sample in np.arange(self.no_samples):
                sum_[sample] = np.sum(pred_cdf_full[sample] <= point) / len(pred_cdf_full[sample])

            kendall_func_point = np.sum(sum_) / self.no_samples
            true_cdf_point = np.sum(true_cdf_full <= point) / self.no_samples
            kendall_calibration[count] = kendall_func_point - true_cdf_point
            count += 1

        np.save(self.path + self.validation_folder + 'kendall_calibration.npy', kendall_calibration)
        print('Completed kendall calibration')

    def _create_templates(self):

        template = []
        template_same = []
        for feature in np.arange(self.no_features):

            if feature != (self.no_features - 1):
                template.append('(pdf[:,' + str(feature) + '] < pdf[pred,' + str(feature) + ']) & ')
                template_same.append('(pdf[:,' + str(feature) + '] == pdf[pred,' + str(feature) + ']) & ')
            else:
                template.append('(pdf[:,' + str(feature) + '] < pdf[pred,' + str(feature) + '])')
                template_same.append('(pdf[:,' + str(feature) + '] == pdf[pred,' + str(feature) + '])')

        template_pred = ''.join(template)
        template_same = ''.join(template_same)
        template_true = template_pred.replace('pdf[pred', 'self.y_test[sample')
        template_true = template_true.replace('<', '<=')

        return template_pred, template_true, template_same
