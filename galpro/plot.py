import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

sns.set_style('white')
sns.set_style('ticks')


def plot_scatter(y_test, y_pred, target_features, path):

    folder = '/point_estimates/plots/'
    if os.path.isdir(path + folder):
        print('Previously saved scatter plots have been overwritten')
    else:
        os.mkdir(path + folder)

    no_features = y_test.shape[1]
    for feature in range(no_features):
        min_, max_ = [np.min(y_test[:, feature]), np.max(y_test[:, feature])]
        sns.scatterplot(x=y_test[:, feature], y=y_pred[:, feature], color='purple', edgecolor='purple',
                        alpha=0.6, marker='.')
        plt.plot([min_, max_], [min_, max_], color='black', linestyle='--', linewidth='1')
        plt.xlim([min_, max_])
        plt.ylim([min_, max_])
        plt.xlabel(target_features[feature])
        plt.ylabel('$' + target_features[feature][1:-1] + '_{ML}$')
        plt.savefig(path + folder + str(feature) + '_scatter.png', bbox_inches='tight', dpi=600)


def plot_posterior(pdfs, target_features, path, y_test=None, y_pred=None):

    # Load points estimates if they exists or create
    y_pred, folder, no_samples, no_features = _check_preds(pdfs=pdfs, y_pred=y_pred, target_features=target_features, path=path)

    if y_test is not None:

        for sample in np.arange(no_samples):
            pdf = np.array(pdfs[sample])
            g = sns.jointplot(x=pdf[:, 0], y=pdf[:, 1], kind="kde", space=0, color="darkorchid", n_levels=10,
                              marginal_kws={'lw': 2, 'color': 'darkorchid', 'shade': True, 'alpha': 0.8})
            g.plot_joint(plt.scatter, color="green", s=15, marker="o", alpha=0.6, edgecolor='black')
            g.ax_joint.collections[0].set_alpha(0)
            g.set_axis_labels(target_features[0], target_features[1])

            true, = plt.plot(y_test[sample, 0], y_test[sample, 1], color='gold', marker='*', markersize=10, linestyle='None',
                             label='True')
            predicted, = plt.plot(y_pred[sample, 0], y_pred[sample, 1], color='white', marker='*', markersize=10, linestyle='None',
                             label='Predicted')

            plt.legend(handles=[true, predicted], facecolor='lightgrey', loc='lower right')
            plt.savefig(path + folder + 'joint_pdf_' + str(sample) + '.png', bbox_inches='tight', dpi=600)
            plt.close()

    else:

        for sample in np.arange(no_samples):
            pdf = np.array(pdfs[sample])
            g = sns.jointplot(x=pdf[:, 0], y=pdf[:, 1], kind="kde", space=0, color="darkorchid", n_levels=10,
                              marginal_kws={'lw': 2, 'color': 'darkorchid', 'shade': True, 'alpha': 0.8})
            g.plot_joint(plt.scatter, color="green", s=15, marker="o", alpha=0.6, edgecolor='black')
            g.ax_joint.collections[0].set_alpha(0)
            g.set_axis_labels(target_features[0], target_features[1])

            predicted, = plt.plot(y_pred[sample, 0], y_pred[sample, 1], color='white', marker='*', markersize=10,
                                  linestyle='None',
                                  label='Predicted')

            plt.legend(handles=[predicted], facecolor='lightgrey', loc='lower right')
            plt.savefig(path + folder + 'joint_pdf_' + str(sample) + '.png', bbox_inches='tight', dpi=600)
            plt.close()


def plot_corner(pdfs, target_features, path, y_test=None, y_pred=None):

    # Load points estimates if they exists or create
    y_pred, folder, no_samples, no_features = _check_preds(pdfs=pdfs, target_features=target_features, path=path)

    no_samples = len(pdfs)
    for sample in np.arange(no_samples):
        pdf = pd.DataFrame(np.array(pdfs[sample]), columns=target_features)
        g = sns.PairGrid(data=pdf, corner=True, despine=True)
        #g = g.map_upper(sns.scatterplot)
        g = g.map_lower(sns.kdeplot, shade=True, color='darkorchid', n_levels=10, shade_lowest=False)
        g = g.map_diag(sns.kdeplot, lw=2, color='darkorchid', shade=True)
        plt.savefig(path + folder + 'corner_plot_' + str(sample) + '.png', bbox_inches='tight', dpi=600)
        plt.close()


def _check_preds(pdfs, y_pred, target_features, path):

    no_samples, no_features = [len(pdfs), len(target_features)]
    folder = '/point_estimates/'

    if y_pred is None:
        # Load point estimates if available
        if os.path.isfile(path + folder + 'point_estimates.npy'):
            y_pred = np.load(path + folder + 'point_estimates.npy')
            print('Previously saved point estimates have been loaded')
        else:
            y_pred = np.empty((no_samples, no_features))
            for sample in np.arange(no_samples):
                y_pred[sample] = np.mean(np.array(pdfs[sample]), axis=0)

    folder = '/posteriors/plots/'
    if os.path.isdir(path + folder):
        print('Previously saved posterior plots have been overwritten')
    else:
        os.mkdir(path + folder)

    return y_pred, folder, no_samples, no_features
