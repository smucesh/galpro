import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import h5py
from .validation import Validation
from .plot import Plot
from .utils import convert_1d_arrays
from .conf import set_rf_params


class Model:
    """
    Top-level class for training or loading a previously saved random forest model.
    Uses the implementation of the random forest algorithm in the scikit-learn library.

    Parameters
    ----------
    model_name: str
        The subfolder into which all the information is stored. To train a new model, specify a unique name or to load a
        previously trained model, specify its model name.

    x_train: array_like
        An array of input features of training galaxies with shape [n, x], where n is the number of galaxies and x is
        the number of input features.

    y_train: array_like
        An array of target features of training galaxies with shape [n, y], where n in the number of galaxies and y is
        the number of target features.

    x_test: array_like
        An array of input features of testing galaxies with the same shape as x_train.

    y_test: array_like, optional
        An array of target features of testing galaxies with the same shape as y_train.

    target_features: list, optional
        A list of variables of target features.

    save_model: bool, optional
        Whether to save model or not.
    """

    def __init__(self, model_name, x_train, y_train, x_test, y_test=None, target_features=None, save_model=False):

        # Initialise arguments
        self.model_name = model_name
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.target_features = target_features
        self.save_model = save_model

        # Get random forest hyperparameters
        self.params = set_rf_params()

        # Creating directory paths
        self.path = os.getcwd() + '/galpro/' + str(model_name) + '/'
        self.point_estimate_folder = 'point_estimates/'
        self.posterior_folder = 'posteriors/'
        self.validation_folder = 'validation/'
        self.plot_folder = 'plots/'

        # Creating target variable names
        self.no_features = y_train.ndim
        if self.target_features is None:
            self.target_features = []
            for feature in np.arange(self.no_features):
                self.target_features.append('var_' + str(feature))

        # Check if model_name exists
        if os.path.isdir(self.path):
            print('Model directory exists.')

            # Try to load the model
            if os.path.isfile(self.path + str(self.model_name) + '.sav'):
                self.model = joblib.load(self.path + str(self.model_name) + '.sav')
                print('Loaded Model.')
            else:
                print('No model found. Please choose a different model_name.')
                exit()

        else:
            # Create the model directory
            if not os.path.isdir(os.getcwd() + '/galpro/'):
                os.mkdir(os.getcwd() + '/galpro/')
            os.mkdir(self.path)
            os.mkdir(self.path + self.point_estimate_folder)
            os.mkdir(self.path + self.posterior_folder)
            os.mkdir(self.path + self.validation_folder)
            os.mkdir(self.path + self.point_estimate_folder + self.plot_folder)
            os.mkdir(self.path + self.posterior_folder + self.plot_folder)
            os.mkdir(self.path + self.validation_folder + self.plot_folder)

            # Train model
            self.model = RandomForestRegressor(**self.params)
            self.model.fit(self.x_train, self.y_train)
            print('Trained model.')

            # Save model to directory
            if save_model:
                model_file = self.model_name + '.sav'
                joblib.dump(self.model, self.path + model_file)
                print('Saved model.')

        # Convert 1d arrays
        if self.no_features == 1:
            self.y_train, self.y_test = convert_1d_arrays(self.y_train, self.y_test)

        # Initialise external classes
        self.plot = Plot(y_test=self.y_test, target_features=self.target_features, path=self.path)
        self.validation = Validation(y_test=self.y_test, target_features=self.target_features, path=self.path)

    def point_estimate(self, make_plots=False):
        """
        Make point predictions on test samples using the trained model.

        Parameters
        ----------
        make_plots: bool, optional
            Whether to make scatter plots or not.
        """

        # Use the model to make predictions on new objects
        y_pred = self.model.predict(self.x_test)

        # Save predictions as numpy arrays
        np.save(self.path + self.point_estimate_folder + 'point_estimates.npy', y_pred)
        print('Saved point estimates. Any previously saved point estimates have been overwritten.')

        # Save plots
        if make_plots:
            self.plot_scatter()

    def posterior(self, make_plots=False):
        """
        Produce posterior probability distributions of test samples using the trained model.

        Parameters
        ----------
        make_plots: bool, optional
            Whether to make posterior PDF plots or not.
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
                values[tree][leafs[sample, tree]].extend(list(self.y_train[sample]))

        for sample in np.arange(np.shape(self.x_test)[0]):
            sample_leafs = self.model.apply(self.x_test[sample].reshape(1, self.model.n_features_))[0]
            sample_posterior = []
            for tree in np.arange(self.model.n_estimators):
                sample_posterior.extend(values[tree][sample_leafs[tree]])
            sample_posterior = np.array(sample_posterior).reshape(-1, 2)
            with h5py.File(self.path + self.posterior_folder + str(sample) + ".h5", 'w') as f:
                f.create_dataset('data', data=sample_posterior)

        print('Saved posteriors. Any previously saved posteriors have been overwritten.')

        if make_plots:
            if len(self.target_features) < 2:
                self.plot_marginal()
            elif len(self.target_features) > 2:
                self.plot_corner()
            else:
                self.plot_joint()

    # External Class functions
    def validate(self, make_plots=False):
        """
        Validate univariate and/or multivariate posterior PDFs generated by the trained model.

        Parameters
        ----------
        make_plots: bool, optional
            Whether to make posterior PDF plots or not.
        """
        return self.validation.validate(make_plots=make_plots)

    def plot_scatter(self, show=False, save=True):
        return self.plot.plot_scatter(show=show, save=save)

    def plot_marginal(self, show=False, save=True):
        return self.plot.plot_marginal(show=show, save=save)

    def plot_joint(self, show=False, save=True):
        return self.plot.plot_joint(show=show, save=save)

    def plot_corner(self, show=False, save=True):
        return self.plot.plot_corner(show=show, save=save)

    def plot_pit(self, show=False, save=True):
        return self.plot.plot_pit(show=show, save=save)

    def plot_coppit(self, show=False, save=True):
        return self.plot.plot_coppit(show=show, save=save)

    def plot_marginal_calibration(self, show=False, save=True):
        return self.plot.plot_marginal_calibration(show=show, save=save)

    def plot_kendall_calibration(self, show=False, save=True):
        return self.plot.plot_kendall_calibration(show=show, save=save)