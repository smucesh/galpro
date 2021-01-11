import seaborn as sns
import matplotlib as mpl

"""Configuration file"""


def set_rf_params():
    """
    All hyperparameters for the random forest algorithm can be tuned here. Galpro uses the implementation of the
    algorithm in the Python machine learning library scikit-learn. For a detailed description of each hyperparameter
    please visit: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html.
    The most important hyperparameters are n_estimators, max_depth and max_features.
    """

    params = {
        'n_estimators': 100,
        'criterion': 'mse',
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'min_weight_fraction_leaf': 0,
        'max_features': 'auto',
        'max_leaf_nodes': None,
        'min_impurity_decrease': 0,
        'bootstrap': True,
        'oob_score': False,
        'n_jobs': -1,
        'random_state': None,
        'verbose': 0,
        'warm_start': False,
        'ccp_alpha': 0,
        'max_samples': None
    }

    return params


def set_plot_params():
    """
    All seaborn and matplotlib plotting aesthetics can be defined here. Please visit:
    https://seaborn.pydata.org/tutorial/aesthetics.html
    https://matplotlib.org/3.3.1/tutorials/introductory/customizing.html
    for details.
    """

    sns.set_style('white')
    sns.set_style('ticks')
    mpl.rcParams['font.family'] = "Helvetica"
    mpl.rcParams['mathtext.fontset'] = 'dejavuserif'
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['savefig.dpi'] = 300
    no_points = 100  # Number of evaluation points for marginal and kendall calibration
    no_bins = 100  # Number of bins for histogram plots

    return no_bins, no_points






