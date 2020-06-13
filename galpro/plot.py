import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_scatter(y_test, y_pred, target_features, path):

    for feature in range(len(target_features)):
        min_, max_ = [np.min(y_test[:, feature]), np.max(y_test[:, feature])]
        sns.scatterplot(x=y_test[:, feature], y=y_pred[:, feature], color='purple', edgecolor='purple',
                        alpha=0.6, marker='.')
        plt.plot([min_, max_], [min_, max_], color='black', linestyle='--', linewidth='1')
        plt.xlim([min_, max_])
        plt.ylim([min_, max_])
        plt.xlabel(target_features[feature])
        plt.ylabel('$' + target_features[feature][1:-1] + '_{ML}$')
        plt.savefig(path + '/point_estimates/' + target_features[feature][1:-1] + '_scatter.png',
                    bbox_inches='tight', dpi=600)
