import os

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import h5py

from galpro.validation import Validation
from galpro.plot import Plot
from galpro.utils import convert_1d_arrays
from galpro.config import *


class Model:
    """
    Top-level class for training or loading a previously saved random forest model.
    Uses the implementation of the random forest algorithm in the scikit-learn library.

    Parameters
    ----------
    x_train: array_like
        An array of input features of training galaxies with shape [n, x], where n is the number of galaxies and x is
        the number of input features.

    y_train: array_like
        An array of target features of training galaxies with shape [n, y], where n in the number of galaxies and y is
        the number of target features.

    target_features: list
        A list of variables of target features.

    model_name: str
        The subfolder into which all the information is stored. To train a new model, specify a unique name or to load a
        previously trained model, specify its model name.

    x_test: array_like, optional
        An array of input features of testing galaxies with the same shape as x_train.

    y_test: array_like, optional
        An array of target features of testing galaxies with the same shape as y_train.
    """

    def __init__(self, x_train, y_train, target_features, model_name, x_test=None, y_test=None):

        # Initialise arguments
        self.x_train = x_train
        self.y_train = y_train
        self.target_features = target_features
        self.model_name = model_name
        self.params = set_hyperparameters()
        self.x_test = x_test
        self.y_test = y_test

        # Creating directory paths
        self.path = os.getcwd() + '/' + str(model_name) + '/'
        self.point_estimate_folder = 'point_estimates/'
        self.posterior_folder = 'posteriors/'
        self.validation_folder = 'validation/'

        # Check if model_name exists
        if os.path.isdir(self.path):
            print('Model exists.')

            # Load the model
            self.model = joblib.load(self.path + str(self.model_name) + '.sav')
            print('Loaded model.')

        else:
            # Create the model directory
            os.mkdir(self.path)
            os.mkdir(self.path + self.point_estimate_folder)
            os.mkdir(self.path + self.posterior_folder)
            os.mkdir(self.path + self.validation_folder)
            os.mkdir(self.path + self.point_estimate_folder + 'plots/')
            os.mkdir(self.path + self.posterior_folder + 'plots/')
            os.mkdir(self.path + self.validation_folder + 'plots/')

            # Train model
            self.model = RandomForestRegressor(**self.params)
            self.model.fit(self.x_train, self.y_train)
            print('Trained model.')

            # Save model to directory
            model_file = self.model_name + '.sav'
            joblib.dump(self.model, self.path + model_file)
            print('Saved model.')

        # Convert 1d arrays
        if len(self.target_features) == 1:
            self.y_train, self.y_test = convert_1d_arrays(self.y_train, self.y_test)

        # Initialise external classes
        self.plot = Plot(y_test=self.y_test, target_features=self.target_features, path=self.path)
        self.validation = Validation(y_test=self.y_test, target_features=self.target_features, path=self.path)

    def point_estimate(self, make_plots=False):
        """
        Make point predictions using the trained model.

        Parameters
        ----------
        make_plots: bool, optional
            Whether to make scatter plots or not.
        """

        if self.x_test is None:
            print('Pass in x_test to make point predictions.')
            exit()

        # Use the model to make predictions on new objects
        preds = self.model.predict(self.x_test)

        # Save predictions as numpy arrays
        np.save(self.path + self.point_estimate_folder + 'point_estimates.npy', preds)
        print('Saved point estimates. Any previously saved point estimates have been overwritten.')

        # Save plots
        if make_plots:
            self.plot_scatter()

    def posterior(self, make_plots=False):
        """
        Generate posteriors using the trained model.

        Parameters
        ----------
        make_plots: bool, optional
            Whether to make posterior plots or not.
        """

        # A numpy array with shape training_samples * n_estimators of leaf numbers in each decision tree
        # associated with training samples.
        leafs = self.model.apply(self.x_train)

        # Create an empty list which is a list of decision trees within which there are all leafs
        # Each leaf is an empty list
        values = [[[] for leaf in np.arange(np.max(leafs) + 1)] for tree in np.arange(self.model.n_estimators)]

        # Go through each training sample and append the redshift or stellar mass values to the
        # empty list depending on which leaf it is associated with.
        for sample in np.arange(self.x_train.shape[0]):
            for tree in np.arange(self.model.n_estimators):
                values[tree][leafs[sample, tree]].append(list(self.y_train[sample]))

        for sample in np.arange(np.shape(self.x_test)[0]):
            sample_leafs = self.model.apply(self.x_test[sample].reshape(1, self.model.n_features_))[0]
            sample_pdf = []
            for tree in np.arange(self.model.n_estimators):
                sample_pdf.extend(values[tree][sample_leafs[tree]])
            f = h5py.File(self.path + self.posterior_folder + str(sample) + ".h5", "w")
            f.create_dataset('data', data=np.array(sample_pdf))

        print('Saved posteriors. Any previously saved posteriors have been overwritten.')

        if make_plots:
            if len(self.target_features) < 2:
                self.plot_marginal()
            elif len(self.target_features) > 2:
                self.plot_corner()
            else:
                self.plot_posterior()

    # External Classes functions
    def validate(self):
        return self.validation.validate()

    def plot_scatter(self):
        return self.plot.plot_scatter()

    def plot_marginal(self):
        return self.plot.plot_marginal()

    def plot_posterior(self):
        return self.plot.plot_posterior()

    def plot_corner(self):
        return self.plot.plot_corner()

    def plot_pit(self):
        return self.plot.plot_pit()

    def plot_coppit(self):
        return self.plot.plot_coppit()

    def plot_marginal_calibration(self):
        return self.plot.plot_marginal_calibration()

    def plot_kendall_calibration(self):
        return self.plot.plot_kendall_calibration()