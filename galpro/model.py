import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import h5py
import os


class Model:

    def __init__(self, x_train, y_train, params, input_features, target_features,
                 model_name=None, model_file=None, save_model=False):

        # Check if model name and model file are both given
        if model_file and model_name is not None:
            print('Please either specify a model_name if training a new model or a model_file '
                  'if loading a trained model')
            exit()

        # Check if model has already been run
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

        if self.model_file is None:
            # Train model
            self.model = RandomForestRegressor(**self.params)
            self.model.fit(self.x_train, self.y_train)

            os.mkdir(str(self.model_name))
            self.path = os.getcwd() + '/' + str(self.model_name) + '/'
            if save_model:
                model_file = str(self.model_name) + '.sav'
                joblib.dump(self.model, self.path + model_file)

        else:
            self.path = os.getcwd() + '/' + str(self.model_file)[0:-4] + '/'

    def point_estimate(self, x_test, y_test, save_preds=False, make_plots=False):

        if self.model_file is not None:
            self.model = joblib.load(self.path + self.model_file)

        preds = self.model.predict(x_test)

        if save_preds:
            np.save(self.path + 'point_estimates.npy', preds)

        print(preds)
        return preds

    def posterior(self, x_test, y_test, save_pdfs=False, make_plots=False):

        if self.model_file is not None:
            self.model = joblib.load(self.path + self.model_file)

        # A numpy array with shape training_samples * n_estimators of leaf numbers in each decision tree
        # associated with training samples.
        leafs = self.model.apply(self.x_train)

        # Create an empty list which is a list of decision trees within which there are all leafs
        # Each leaf is an empty list
        values = [[[] for leaf in range(np.max(leafs) + 1)] for tree in range(self.model.n_estimators)]

        # Go through each training sample and append the redshift or stellar mass values to the
        # empty list depending on which leaf it is associated with.
        for sample in range(self.x_train.shape[0]):
            for tree in range(self.model.n_estimators):
                values[tree][leafs[sample, tree]].append(list(self.y_train[sample]))

        pdfs = [[] for sample in range(np.shape(x_test)[0])]
        for sample in range(np.shape(x_test)[0]):
            sample_leafs = self.model.apply(x_test[sample].reshape(1, self.model.n_features_))[0]
            sample_pdf = []
            for tree in range(self.model.n_estimators):
                sample_pdf.extend(values[tree][sample_leafs[tree]])
            pdfs[sample].extend(sample_pdf)

        if save_pdfs:
            if not os.path.isdir('model/posterior'):
                os.mkdir('model/posterior')
            for sample in range(10):
                sample_pdf = np.array(pdfs[sample])
                f = h5py.File(self.path + 'posterior/' + str(sample) + ".h5", "w")
                f.create_dataset('data', data=sample_pdf)

        return pdfs
