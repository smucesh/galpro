import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import h5py
import os


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
            self.path = os.getcwd() + '/' + str(self.model_name) + '/'
            if save_model:
                model_file = str(self.model_name) + '.sav'
                joblib.dump(self.model, self.path + model_file)

        else:
            self.path = os.getcwd() + '/' + str(self.model_file)[0:-4] + '/'

    def point_estimate(self, x_test, y_test, run_metrics=False, save_preds=False, make_plots=False):

        if self.model_file is not None:
            self.model = joblib.load(self.path + self.model_file)

        self.y_preds = self.model.predict(x_test)

        if save_preds:
            if os.path.isfile('model/point_estimates.npy'):
                print('Previously saved point estimates have been overwritten')
            np.save(self.path + 'point_estimates.npy', self.y_preds)

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
            if os.path.isdir('model/posterior'):
                print('Previously saved posteriors have been overwritten')
            else:
                os.mkdir('model/posterior')
            for sample in np.arange(y_test.shape[0]):
                sample_pdf = np.array(self.pdfs[sample])
                f = h5py.File(self.path + 'posterior/' + str(sample) + ".h5", "w")
                f.create_dataset('data', data=sample_pdf)

        return self.pdfs

    def validate(self, y_test, make_plots=False, save_validation=False):

        no_samples, no_features = [y_test.shape[0], y_test.shape[1]]

        # Check if posterior() has been run
        if self.pdfs is None:
            # If not then try to load any saved posteriors
            self.pdfs = []
            if os.path.isdir('model/posterior'):
                for sample in np.arange(no_samples):
                    pdf = h5py.File(self.path + 'posterior/' + str(sample) + ".h5", "r")
                    self.pdfs.append(pdf['data'][:])
            else:
                print('No posteriors have been found. Please run Model.posterior()'
                      'to generate posteriors and then run validate().')

        # Probabilistic calibration
        pits = np.empty((no_samples, no_features))
        marginal_outliers = np.empty(no_features)

        for feature in np.arange(no_features):
            for sample in np.arange(no_samples):
                pdf = np.array(self.pdfs[sample])
                pits[sample] = np.sum(pdf[:, feature] <= y_test[sample, feature])/pdf.shape[0]

            no_outliers = np.count_nonzero(pits[:, feature] == 0) + np.count_nonzero(pits[:, feature] == 1)
            marginal_outliers[feature] = (no_outliers/no_samples)*100

        # Marginal calibration
        no_points = 100
        marginal_calibration = np.empty((no_points, no_features))

        for feature in np.arange(no_features):
            count = 0
            min_, max_ = [np.int(np.min(y_test[:, feature])), np.int(np.max(y_test[:, feature]))]

            for point in np.linspace(min_, max_, no_points):
                sum_ = np.zeros(no_samples)
                for sample in np.arange(no_samples):
                    pdf = np.array(self.pdfs[sample])
                    sum_[sample] = np.sum(pdf[:, feature] <= point)/pdf.shape[0]

                pred_cdf = np.sum(sum_)/y_test.shape[0]
                true_cdf = np.sum(y_test[:, feature] <= point)/y_test.shape[0]
                marginal_calibration[count, feature] = pred_cdf - true_cdf
                count += 1

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

        # Probabilistic copula calibration

        # Creating a list of list containing pred_cdf of each point in predictions
        pred_cdf_full = [[] for i in np.arange(no_samples)]
        true_cdf_full = []
        coppits = np.empty(no_samples)
        template_pred, template_true = _create_templates(no_features)

        for sample in np.arange(no_samples):
            pdf = np.array(self.pdfs[sample])
            no_preds = pdf.shape[0]

            for pred in np.arange(no_preds):
                # For point at edges, if <= used, then point counts and cdf is never 0.
                # If <= is used, a large number of point will have near 0 probability, as a result, there will
                # be a peak at 0.
                # -1 insures, the point in consideration does not count when determining cdf.

                pred_cdf_full[sample].append(np.sum(eval(template_pred))/(no_preds-1))

            true_cdf_full.append(np.sum(eval(template_true))/no_preds)
            coppits[sample] = np.sum(pred_cdf_full[sample] <= true_cdf_full[sample])/no_preds

        # Kendall calibration
        kendall_calibration = np.empty(no_points)
        count = 0

        for point in np.linspace(0, 1, no_points):
            sum_ = np.zeros(no_samples)
            for sample in np.arange(no_samples):
                sum_[sample] = np.sum(pred_cdf_full[sample] <= point)/len(pred_cdf_full[sample])

            kendall_func_point = np.sum(sum_)/no_samples
            true_cdf_point = np.sum(true_cdf_full <= point)/no_samples
            kendall_calibration[count] = kendall_func_point - true_cdf_point
            count += 1

        return kendall_calibration





