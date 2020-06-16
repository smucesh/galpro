import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import h5py
import os
from galpro.validation import validate
from galpro.plot import *


class Model:

    def __init__(self, x_train, y_train, params, target_features, input_features=None,
                 model_name=None, model_file=None, save_model=False):

        # Check if model_name and model_file are both given
        if model_file and model_name is not None:
            print('Please either specify a model_name if training a new model or a model_file '
                  'if loading a trained model')
            exit()

        # Check if model_name exists
        if os.path.isdir(str(model_name)):
            print('The model with the specified name already exists. '
                  'Please choose a different model_name or delete the model directory.')
            exit()

        # Initialise
        self.x_train = x_train
        self.y_train = y_train
        self.params = params
        self.input_features = input_features
        self.target_features = target_features
        self.model_name = model_name
        self.model_file = model_file
        self.save_model = save_model
        self.preds = None
        self.pdfs = None

        if self.model_file is None:
            # Train model
            self.model = RandomForestRegressor(**self.params)
            self.model.fit(self.x_train, self.y_train)

            os.mkdir(str(self.model_name))
            self.path = os.getcwd() + '/' + self.model_name
            if save_model:
                model_file = self.model_name + '.sav'
                joblib.dump(self.model, self.path + model_file)

        else:
            self.model_name = str(self.model_file)[0:-4]
            self.path = os.getcwd() + '/' + self.model_name

    def point_estimate(self, x_test, y_test=None, save_preds=False, make_plots=False):

        folder = '/point_estimates/'
        if self.model_file is not None:
            self.model = joblib.load(self.path + '/' + self.model_file)

        self.preds = self.model.predict(x_test)

        if save_preds:
            if os.path.isdir(self.path + folder):
                print('Previously saved point estimates have been overwritten')
            else:
                os.mkdir(self.path + folder)
            np.save(self.path + folder + 'point_estimates.npy', self.preds)

        if make_plots:
            if y_test is not None:
                self.plot_scatter(y_test=y_test, y_pred=self.preds)
            else:
                print('Scatter plots cannot be made as y_test is not available for comparison')

        return self.preds

    def posterior(self, x_test, y_test=None, save_pdfs=False, make_plots=False):

        if self.model_file is not None:
            self.model = joblib.load(self.path + '/' + self.model_file)

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

        self.pdfs = [[] for sample in np.arange(np.shape(x_test)[0])]
        for sample in np.arange(np.shape(x_test)[0]):
            sample_leafs = self.model.apply(x_test[sample].reshape(1, self.model.n_features_))[0]
            sample_pdf = []
            for tree in np.arange(self.model.n_estimators):
                sample_pdf.extend(values[tree][sample_leafs[tree]])
            self.pdfs[sample].extend(sample_pdf)

        folder = '/posteriors/'
        if save_pdfs:
            if os.path.isdir(self.path + folder):
                print('Previously saved posteriors have been overwritten')
            else:
                os.mkdir(self.path + folder)
            for sample in np.arange(x_test.shape[0]):
                sample_pdf = np.array(self.pdfs[sample])
                f = h5py.File(self.path + folder + str(sample) + ".h5", "w")
                f.create_dataset('data', data=sample_pdf)

        if make_plots:
            if len(self.target_features) > 2:
                self.plot_corner(pdfs=self.pdfs, y_test=y_test, y_pred=self.preds)
            else:
                self.plot_posterior(pdfs=self.pdfs, y_test=y_test, y_pred=self.preds)

        return self.pdfs

    def validate(self, y_test, save_validation=False, make_plots=False):
        return validate(y_test=y_test, make_plots=make_plots, save_validation=save_validation,
                        pdfs=self.pdfs, path=self.path)

    def plot_scatter(self, y_test, y_pred):
        return plot_scatter(y_test=y_test, y_pred=y_pred, target_features=self.target_features, path=self.path)

    def plot_posterior(self, pdfs, y_test=None, y_pred=None):
        return plot_posterior(pdfs=pdfs, y_test=y_test, y_pred=y_pred, target_features=self.target_features, path=self.path)

    def plot_corner(self, pdfs, y_test=None, y_pred=None):
        return plot_corner(pdfs=pdfs, y_test=y_test, y_pred=y_pred, target_features=self.target_features, path=self.path)
