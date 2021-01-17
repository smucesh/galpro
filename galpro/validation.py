import os
import numpy as np
import h5py
from .plot import Plot
from .utils import load_posteriors, create_templates
from .conf import set_plot_params


class Validation:
    """
    Internal class for performing validation on the univariate and/or multivariate posterior PDFs of the test samples
    generated by the trained model.

    The marginal PDFs are validated using the framework developed by Gneiting et al. (2007)
    (https://hal.archives-ouvertes.fr/file/index/docid/363242/filename/jrssb1b.pdf).
    The multivariate PDFs are validated using the multivariate extension of the framework developed by
    Ziegel and Gneiting. (2014) (https://projecteuclid.org/download/pdfview_1/euclid.ejs/1418313582).

    Parameters
    ----------
    y_test: array_like
        An array of target features of testing galaxies with the same shape as y_train.

    target_features: list
        A list of variables of target features.

    path: str
        Location of the model directory.
    """

    def __init__(self, y_test, y_pred, posteriors, validation, target_features, no_samples, no_features, path):

        # Initialise arguments
        self.y_test = y_test
        self.y_pred = y_pred
        self.posteriors = posteriors
        self.validation = validation
        self.target_features = target_features
        self.no_samples = no_samples
        self.no_features = no_features
        self.path = path

        # Internal parameters
        self.no_points = set_plot_params()[0]
        self.posterior_folder = 'posteriors/'
        self.validation_folder = 'validation/'

        # Initialise classes
        self.plot = Plot(y_test=self.y_test, y_pred=self.y_pred, posteriors=self.posteriors, validation=self.validation,
                         target_features=self.target_features, no_samples=self.no_samples, no_features=self.no_features,
                         path=self.path)

    def validate(self, save_validation=False, make_plots=False):
        """Top-level function for performing all modes of validation."""

        if self.y_test is None:
            print('Provide y_test to perform validation.')
            exit()

        # Load posteriors
        if self.posteriors is None:
            self.posteriors = load_posteriors(path=self.path)

        # Validation file
        validation = h5py.File(self.path + self.validation_folder + "validation.h5", 'w', driver='core',
                               backing_store=save_validation)

        # Run validation
        pits = self.probabilistic_calibration()
        marginal_calibration = self.marginal_calibration()
        validation.create_dataset('pits', data=pits)
        validation.create_dataset('marginal_calibration', data=marginal_calibration)

        if self.no_features > 1:
            coppits, pred_cdf_full, true_cdf_full = self.probabilistic_copula_calibration()
            kendall_calibration = self.kendall_calibration(pred_cdf_full, true_cdf_full)
            validation.create_dataset('coppits', data=coppits)
            validation.create_dataset('kendall_calibration', data=kendall_calibration)

        # Create plots
        self.plot.validation = validation
        if make_plots:
            print('Creating PIT plots.')
            self.plot.plot_pit()
            print('Creating marginal calibration plots.')
            self.plot.plot_marginal_calibration()
            if self.no_features > 1:
                print('Creating copPIT plots.')
                self.plot.plot_coppit()
                print('Creating kendall calibration plots.')
                self.plot.plot_kendall_calibration()

        if save_validation:
            print('Saved validation. Any previously saved validation has been overwritten.')

        return validation

    def probabilistic_calibration(self):
        """Performs probabilistic calibration"""

        pits = np.empty((self.no_samples, self.no_features))

        for feature in np.arange(self.no_features):
            for sample in np.arange(self.no_samples):
                posterior = self.posteriors[str(sample)][:]
                pits[sample, feature] = np.sum(posterior[:, feature] <= self.y_test[sample, feature]) / posterior.shape[0]

        print('Completed probabilistic calibration.')
        return pits

    def marginal_calibration(self):
        """Performs marginal calibration"""

        marginal_calibration = np.empty((self.no_points, self.no_features))

        for feature in np.arange(self.no_features):
            count = 0
            min_, max_ = [np.floor(np.min(self.y_test[:, feature])), np.ceil(np.max(self.y_test[:, feature]))]

            for point in np.linspace(min_, max_, self.no_points):
                sum_ = np.zeros(self.no_samples)
                for sample in np.arange(self.no_samples):
                    posterior = self.posteriors[str(sample)][:]
                    sum_[sample] = np.sum(posterior[:, feature] <= point) / posterior.shape[0]

                pred_cdf_marg_point = np.sum(sum_) / self.no_samples
                true_cdf_marg_point = np.sum(self.y_test[:, feature] <= point) / self.no_samples
                marginal_calibration[count, feature] = pred_cdf_marg_point - true_cdf_marg_point
                count += 1

        print('Completed marginal calibration.')
        return marginal_calibration

    def probabilistic_copula_calibration(self):
        """Performs probabilistic copula calibration"""

        # Creating a list of list containing pred_cdf of each point in predictions
        pred_cdf_full = [[] for i in np.arange(self.no_samples)]
        true_cdf_full = []
        coppits = np.empty(self.no_samples)
        template_pred, template_true, template_same = create_templates(no_features=self.no_features)

        for sample in np.arange(self.no_samples):
            posterior = self.posteriors[str(sample)][:]
            no_preds = posterior.shape[0]

            for pred in np.arange(no_preds):
                # For point at edges, if <= used, then point counts and cdf is never 0.
                # If <= is used, a large number of point will have near 0 probability, as a result, there will
                # be a peak at 0.
                # -1 insures, the point in consideration does not count when determining cdf.
                same_preds = np.sum(eval(template_same))
                pred_cdf_full[sample].append(np.sum(eval(template_pred)) / (no_preds - same_preds))

            true_cdf_full.append(np.sum(eval(template_true)) / no_preds)
            coppits[sample] = np.sum(pred_cdf_full[sample] <= true_cdf_full[sample]) / no_preds

        print('Completed probabilistic copula calibration.')
        return coppits, pred_cdf_full, true_cdf_full

    def kendall_calibration(self, pred_cdf_full, true_cdf_full):
        """Performs kendall calibration"""

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

        print('Completed kendall calibration.')
        return kendall_calibration
