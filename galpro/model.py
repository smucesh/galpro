import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import h5py
import os
from galpro.validation import Validation
from galpro.plot import Plot


class Model:

    def __init__(self, x_train, y_train, params, target_features, input_features=None,
                 model_name=None, save_model=False):

        # Initialise arguments
        self.x_train = x_train
        self.y_train = y_train
        self.params = params
        self.input_features = input_features
        self.target_features = target_features
        self.model_name = model_name
        self.save_model = save_model
        self.preds = None
        self.pdfs = None
        self.metrics = None
        self.path = os.getcwd() + '/' + str(model_name) + '/'

        # Check if model_name exists
        if os.path.isdir(self.path):
            print('The model name already exists.')

            # Check if the model is saved
            if os.path.isfile(self.path + str(self.model_name) + '.sav'):

                # Load the model
                self.model = joblib.load(self.path + str(self.model_name) + '.sav')
                print('The model file has been found and it has been loaded.')

            else:
                print('The model file has not been found.'
                      'Please choose a different model_name or delete the model directory.')
                exit()

        else:
            # Create the model directory
            os.mkdir(str(self.model_name))

            # Train model
            self.model = RandomForestRegressor(**self.params)
            self.model.fit(self.x_train, self.y_train)
            print('Training the model...')

            # Save model to directory
            if save_model:
                model_file = self.model_name + '.sav'
                joblib.dump(self.model, self.path + model_file)
                print('The model has been saved.')
                exit()

        # Initialise classes
        self.plot = Plot(target_features=self.target_features, path=self.path)
        self.validation = Validation(target_features=self.target_features, path=self.path)

    def point_estimate(self, x_test, y_test=None, save_preds=False, make_plots=False):

        # Use the model to make predictions on new objects
        self.preds = self.model.predict(x_test)

        folder = '/point_estimates/'
        # Save predictions as numpy arrays
        if save_preds:
            if os.path.isdir(self.path + folder):
                print('Previously saved point estimates have been overwritten')
            else:
                os.mkdir(self.path + folder)
            np.save(self.path + folder + 'point_estimates.npy', self.preds)

        # Save plots
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
                #self.plot_corner(pdfs=self.pdfs, y_test=y_test, y_pred=self.preds)
                self.plot_posterior(pdfs=self.pdfs, y_test=y_test, y_pred=self.preds)

        return self.pdfs

    # External Classes functions
    def validate(self, y_test, pdfs=None, save_validation=False, make_plots=False):
        return self.validation.validate(y_test=y_test, pdfs=pdfs, make_plots=make_plots, save_validation=save_validation)

    def plot_scatter(self, y_test, y_pred):
        return self.plot.plot_scatter(y_test=y_test, y_pred=y_pred)

    def plot_posterior(self, pdfs, y_test=None, y_pred=None):
        return self.plot.plot_posterior(pdfs=pdfs, y_test=y_test, y_pred=y_pred)

    def plot_corner(self, pdfs, y_test=None, y_pred=None):
        return self.plot.plot_corner(pdfs=pdfs, y_test=y_test, y_pred=y_pred)

    def plot_pit(self, pit):
        return self.plot.plot_pit(pit=pit)

    def plot_coppit(self, coppit):
        return self.plot.plot_coppit(coppit=coppit)

    def plot_marginal_calibration(self, marginal_calibration, y_test):
        return self.plot.plot_marginal_calibration(marginal_calibration=marginal_calibration, y_test=y_test)

    def plot_kendall_calibration(self, kendall_calibration):
        return self.plot.plot_kendall_calibration(kendall_calibration=kendall_calibration)



