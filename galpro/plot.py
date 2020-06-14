import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


def plot_scatter(y_test, y_pred, target_features, path, model_name):
    sns.set_style('white')
    sns.set_style("ticks")

    if os.path.isdir(model_name + '/point_estimates/plots'):
        print('Previously saved scatter plots have been overwritten')
    else:
        os.mkdir(model_name + '/point_estimates/plots')

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
        plt.savefig(path + '/point_estimates/plots/' + target_features[feature][1:-1] + '_scatter.png',
                    bbox_inches='tight', dpi=600)


def plot_posterior(y_test, y_pred, pdfs, target_features, path, model_name):
    sns.set_style('white')
    sns.set_style("ticks")

    if os.path.isdir(model_name + '/posterior/plots'):
        print('Previously saved posterior plots have been overwritten')
    else:
        os.mkdir(model_name + '/posterior/plots')

    no_samples = y_test.shape[0]
    for sample in np.arange(no_samples):
        pdf = np.array(pdfs[sample])
        g = sns.jointplot(x=pdf[:, 0], y=pdf[:, 1], kind="kde", space=0, color="darkorchid", n_levels=10,
                          marginal_kws={'lw': 2, 'color': 'darkorchid', 'shade': True, 'alpha': 0.8})
        g.plot_joint(plt.scatter, color="green", s=15, marker="o", alpha=0.6, edgecolor='black')
        g.ax_joint.collections[0].set_alpha(0)
        g.set_axis_labels(r'$' + target_features[0][1:-1] + '$', r'$' + target_features[1][1:-1] + '$')

        true, = plt.plot(y_test[sample, 0], y_test[sample, 1], color='gold', marker='*', markersize=10, linestyle='None',
                         label='True')
        predicted, = plt.plot(y_pred[sample, 0], y_pred[sample, 1], color='white', marker='*', markersize=10, linestyle='None',
                         label='Predicted')

        plt.legend(handles=[true, predicted], facecolor='lightgrey', loc='lower right')
        plt.savefig(path + '/posterior/plots/' + 'joint_pdf_' + str(sample) + '.png',
                    bbox_inches='tight', dpi=600)
        plt.close()


def plot_corner(y_test, y_pred, pdfs, target_features, path, model_name):
    sns.set_style('white')
    sns.set_style("ticks")

    if os.path.isdir(model_name + '/posterior/plots'):
        print('Previously saved corner plots have been overwritten')
    else:
        os.mkdir(model_name + '/posterior/plots')

    no_samples = y_test.shape[0]
    for sample in np.arange(no_samples):
        pdf = pd.DataFrame(np.array(pdfs[sample]), columns=['$z$', '$M_{\star}$'])
        g = sns.PairGrid(data=pdf, corner=True, despine=True)
        g = g.map_lower(sns.kdeplot, shade=True, color='darkorchid', n_levels=10, shade_lowest=False)
        g = g.map_diag(sns.kdeplot, lw=2, color='darkorchid', shade=True)
        plt.savefig(path + '/posterior/plots/' + 'corner_plot_' + str(sample) + '.png', bbox_inches='tight', dpi=600)
        plt.close()