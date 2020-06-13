import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import h5py
import os
from galpro.validation import validate
from galpro.plot import plot_scatter


class Model:

    def __init__(self, x_train, y_train, params, input_features, target_features,
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
        self.y_preds = None
        self.pdfs = None

        if self.model_file is None:
            # Train model
            self.model = RandomForestRegressor(**self.params)
            self.model.fit(self.x_train, self.y_train)

            os.mkdir(str(self.model_name))
            self.path = os.getcwd() + '/' + self.model_name + '/'
            if save_model:
                model_file = self.model_name + '.sav'
                joblib.dump(self.model, self.path + model_file)

        else:
            self.model_name = str(self.model_file)[0:-4]
            self.path = os.getcwd() + '/' + self.model_name

    def point_estimate(self, x_test, y_test, run_metrics=False, save_preds=False, make_plots=False):

        if self.model_file is not None:
            self.model = joblib.load(self.path + '/' +self.model_file)

        self.y_preds = self.model.predict(x_test)

        if save_preds:
            if os.path.isdir(self.model_name + '/point_estimates'):
                print('Previously saved point estimates have been overwritten')
            else:
                os.mkdir(self.model_name + '/point_estimates')
            np.save(self.path + '/point_estimates/' + 'point_estimates.npy', self.y_preds)

        if make_plots:
            if os.path.isdir(self.model_name + '/point_estimates'):
                print('Previously saved scatter plots have been overwritten')
            else:
                os.mkdir(self.model_name + '/point_estimates')
            self.plot_scatter(y_test=y_test, y_pred=self.y_preds, target_features=self.target_features)

        return self.y_preds

    def posterior(self, x_test, y_test, save_pdfs=False, make_plots=False):

        if self.model_file is not None:
            self.model = joblib.load(self.path + self.model_file)

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

        if save_pdfs:
            if os.path.isdir(self.model_name + '/posterior'):
                print('Previously saved posteriors have been overwritten')
            else:
                os.mkdir(self.model_name + '/posterior')
            for sample in np.arange(y_test.shape[0]):
                sample_pdf = np.array(self.pdfs[sample])
                f = h5py.File(self.path + '/posterior/' + str(sample) + ".h5", "w")
                f.create_dataset('data', data=sample_pdf)

        return self.pdfs

    def validate(self, y_test, save_validation=False, make_plots=False):
        return validate(y_test=y_test, make_plots=make_plots, save_validation=save_validation,
                        pdfs=self.pdfs, path=self.path, model_name=self.model_name)

    def plot_scatter(self, y_test, y_pred, target_features):
        return plot_scatter(y_test=y_test, y_pred=y_pred, target_features=target_features, path=self.path)