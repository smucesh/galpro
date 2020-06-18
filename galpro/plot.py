import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import statsmodels.api as sm

from galpro.metrics import Metrics


class Plot:

    def __init__(self, target_features, path):

        # Initialise arguments
        self.target_features = target_features
        self.path = path
        self.no_features = len(target_features)
        self.point_estimate_folder = '/point_estimates/plots/'
        self.posterior_folder = '/posteriors/plots/'
        self.validation_folder = '/validation/plots/'

        # Initialise class
        self.metrics = Metrics()

        # Set seaborn and matplotlib plot settings
        sns.set_style('white')
        sns.set_style('ticks')
        mpl.rcParams['font.family'] = "Helvetica"
        mpl.rcParams['mathtext.fontset'] = 'dejavuserif'
        mpl.rcParams['font.size'] = 12

    def plot_scatter(self, y_test, y_pred):

        # Check if folder already exists
        self._check_folder_exists(folder=self.point_estimate_folder)

        # Get metrics
        metrics = self.metrics.pred_metrics(y_test=y_test, y_pred=y_pred)

        for feature in np.arange(self.no_features):
            min_, max_ = [np.min(y_test[:, feature]), np.max(y_test[:, feature])]
            sns.scatterplot(x=y_test[:, feature], y=y_pred[:, feature], color='purple', edgecolor='purple',
                            alpha=0.6, marker='.')
            plt.plot([min_, max_], [min_, max_], color='black', linestyle='--', linewidth='1')
            plt.plot([], [], ' ', label=f'RMSE: {metrics[feature]}')
            plt.xlim([min_, max_])
            plt.ylim([min_, max_])
            plt.xlabel(self.target_features[feature])
            plt.ylabel('$' + self.target_features[feature][1:-1] + '_{ML}$')
            plt.legend(edgecolor='None', loc='lower right')
            plt.savefig(self.path + self.point_estimate_folder + str(feature) + '_scatter.png', bbox_inches='tight',
                        dpi=600)
            plt.close()

    def plot_posterior(self, pdfs, y_test=None, y_pred=None):

        max_features = 2
        if len(self.target_features) != max_features:
            print('Number of target features is greater than 2. As a result, plot_posterior cannot run.'
                  'Please run plot_corner instead.')
            exit()

        # Load points estimates if they exists
        y_pred = self._check_preds(pdfs=pdfs, y_pred=y_pred)

        # Check if folder already exists
        self._check_folder_exists(folder=self.posterior_folder)

        no_samples = len(pdfs)

        for sample in np.arange(no_samples):
            pdf = np.array(pdfs[sample])
            g = sns.jointplot(x=pdf[:, 0], y=pdf[:, 1], kind="kde", space=0, color="darkorchid", n_levels=10,
                              marginal_kws={'lw': 2, 'color': 'darkorchid', 'shade': True, 'alpha': 0.8})
            g.plot_joint(plt.scatter, color="green", s=15, marker="o", alpha=0.6, edgecolor='black')
            g.ax_joint.collections[0].set_alpha(0)
            g.set_axis_labels(self.target_features[0], self.target_features[1])
            predicted, = plt.plot(y_pred[sample, 0], y_pred[sample, 1], color='white', marker='*', markersize=10,
                                  linestyle='None', label='Predicted')

            if y_test is not None:
                true, = plt.plot(y_test[sample, 0], y_test[sample, 1], color='gold', marker='*', markersize=10,
                                 linestyle='None', label='True')
                plt.legend(handles=[true, predicted], facecolor='lightgrey', loc='lower right')
            else:
                plt.legend(handles=[predicted], facecolor='lightgrey', loc='lower right')

            plt.savefig(self.path + self.posterior_folder + 'joint_pdf_' + str(sample) + '.png', bbox_inches='tight',
                        dpi=600)
            plt.close()

    def plot_corner(self, pdfs, y_test=None, y_pred=None):

        # Check if folder already exists
        self._check_folder_exists(folder=self.posterior_folder)

        # Get quantiles
        quantiles = self.metrics.quantiles(pdfs=pdfs)

        no_samples = len(pdfs)
        for sample in np.arange(no_samples):
            pdf = pd.DataFrame(np.array(pdfs[sample]), columns=self.target_features)
            g = sns.PairGrid(data=pdf, corner=True)
            #g = g.map_upper(sns.scatterplot)
            g = g.map_lower(sns.kdeplot, shade=True, color='darkorchid', n_levels=10, shade_lowest=False)
            g = g.map_diag(sns.kdeplot, lw=2, color='darkorchid', shade=True)

            for feature in np.arange(self.no_features):
                g.axes[feature, feature].set_title(f'{self.target_features[feature]}'
                                                   f'$= {quantiles[feature, sample, 1]:.2f}'
                                                   f'^{{+{quantiles[feature, sample, 2]-quantiles[feature, sample, 1]:.2f}}}'
                                                   f'_{{-{quantiles[feature, sample, 1]-quantiles[feature, sample, 0]:.2f}}}$',
                                                   fontsize=10)
                g.axes[feature, feature].axvline(quantiles[feature, sample, 0], color='black', linestyle='--', linewidth=1)
                g.axes[feature, feature].axvline(quantiles[feature, sample, 1], color='black', linestyle='--', linewidth=1)
                g.axes[feature, feature].axvline(quantiles[feature, sample, 2], color='black', linestyle='--', linewidth=1)

            sns.despine(top=False, left=False, right=False, bottom=False)
            plt.savefig(self.path + self.posterior_folder + 'corner_plot_' + str(sample) + '.png', bbox_inches='tight',
                        dpi=600)
            plt.close()

    def plot_pit(self, pit):

        # Check if folder already exists
        self._check_folder_exists(folder=self.validation_folder)

        # Get marginal pdf metrics
        outliers, kld, kst, cvm = self.metrics.pdf_metrics(data=pit, no_features=2)

        for feature in np.arange(self.no_features):
            qqplot = sm.qqplot(pit[:, feature], 'uniform', line='45').gca().lines
            qq_theory, qq_data = [qqplot[0].get_xdata(), qqplot[0].get_ydata()]
            plt.close()

            ax1 = sns.distplot(pit[:, feature], bins=100, kde=False,
                               hist_kws={'color': 'slategrey', 'edgecolor': 'None', 'alpha': 0.5})
            ax2 = plt.twinx()
            #ax2 = sns.scatterplot(x=qq_theory, y=qq_data)
            ax2 = sns.lineplot(x=qq_theory, y=qq_data, color='blue')
            ax2.plot([0, 1], [0, 1], color='black', linewidth=1, linestyle='--')
            plt.plot([], [], ' ', label=f'Outliers: {outliers[feature]:.2f}%')
            plt.plot([], [], ' ', label=f'KLD: {kld[feature]:.2f}')
            plt.plot([], [], ' ', label=f'KST: {kst[feature]:.2f}')
            plt.plot([], [], ' ', label=f'CvM: {cvm[feature]:.2f}')

            ax1.set_xlabel('$Q_{theory}/PIT$')
            ax1.set_ylabel('$N$')
            ax2.set_ylabel('$Q_{data}$')
            ax2.set_xlim([0, 1])
            ax2.set_ylim([0, 1])
            plt.legend(framealpha=0, edgecolor='None', loc='lower right')
            plt.savefig(self.path + self.validation_folder + str(feature) + '_pit.png', bbox_inches='tight', dpi=600)
            plt.close()

    def plot_coppit(self, coppit):

        # Check if folder already exists
        self._check_folder_exists(folder=self.validation_folder)

        # Get full pdf metrics
        outliers, kld, kst, cvm = self.metrics.pdf_metrics(data=coppit, no_features=1)

        qqplot = sm.qqplot(coppit, 'uniform', line='45').gca().lines
        qq_theory, qq_data = [qqplot[0].get_xdata(), qqplot[0].get_ydata()]
        plt.close()

        ax1 = sns.distplot(coppit, bins=100, kde=False, hist_kws={'color': 'slategrey', 'edgecolor': 'None', 'alpha': 0.5})
        ax2 = plt.twinx()
        #ax2 = sns.scatterplot(x=qq_theory, y=qq_data)
        ax2 = sns.lineplot(x=qq_theory, y=qq_data, color='blue')
        ax2.plot([0, 1], [0, 1], color='black', linewidth=1, linestyle='--')
        plt.plot([], [], ' ', label=f'Outliers: {outliers[0]:.2f}%')
        plt.plot([], [], ' ', label=f'KLD: {kld[0]:.2f}')
        plt.plot([], [], ' ', label=f'KST: {kst[0]:.2f}')
        plt.plot([], [], ' ', label=f'CvM: {cvm[0]:.2f}')

        ax1.set_xlabel('$Q_{theory}/copPIT$')
        ax1.set_ylabel('$N$')
        ax2.set_ylabel('$Q_{data}$')
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        plt.legend(framealpha=0, edgecolor='None', loc='lower right')
        plt.savefig(self.path + self.validation_folder + 'coppit.png', bbox_inches='tight', dpi=600)
        plt.close()

    def plot_marginal_calibration(self, marginal_calibration, y_test):

        # Check if folder already exists
        self._check_folder_exists(folder=self.validation_folder)

        for feature in np.arange(self.no_features):
            min_, max_ = [np.min(y_test[:, feature]), np.max(y_test[:, feature])]
            sns.lineplot(x=np.linspace(min_, max_, 100), y=marginal_calibration[:, feature], color="blue")
            plt.axhline(0, color='black', linewidth=1, linestyle='--')
            plt.ylim([-np.max(marginal_calibration), np.max(marginal_calibration)])
            plt.xlabel(self.target_features[feature])
            plt.ylabel(r'$\hat{F}_{I} - \tilde{G}_{I}$')
            plt.savefig(self.path + self.validation_folder + str(feature) + '_marginal_calibration.png',
                        bbox_inches='tight', dpi=600)
            plt.close()

    def plot_kendall_calibration(self, kendall_calibration):

        # Check if folder already exists
        self._check_folder_exists(folder=self.validation_folder)

        sns.lineplot(x=np.linspace(0, 1, 100), y=kendall_calibration, color="blue")
        plt.axhline(0, color='black', linewidth=1, linestyle='--')
        plt.ylim([-np.max(kendall_calibration), np.max(kendall_calibration)])
        plt.xlabel('$w$')
        plt.ylabel(r'$\mathcal{\hat{K}}_{H_{I}}  - \tilde{J}_{I}$')
        plt.savefig(self.path + self.validation_folder + 'kendall_calibration.png', bbox_inches='tight', dpi=600)
        plt.close()

    def _check_preds(self, pdfs, y_pred):

        no_samples, no_features = [len(pdfs), self.no_features]

        if y_pred is None:
            # Load point estimates if available
            if os.path.isfile(self.path + self.point_estimate_folder + 'point_estimates.npy'):
                y_pred = np.load(self.path + self.point_estimate_folder + 'point_estimates.npy')
                print('Previously saved point estimates have been loaded')
            else:
                y_pred = np.empty((no_samples, no_features))
                for sample in np.arange(no_samples):
                    y_pred[sample] = np.mean(np.array(pdfs[sample]), axis=0)

        return y_pred

    def _check_folder_exists(self, folder):

        if os.path.isdir(self.path + folder):
            if folder == self.point_estimate_folder:
                print('Previously saved scatter plots have been overwritten')
            elif folder == self.posterior_folder:
                print('Previously saved posterior plots have been overwritten')
            elif folder == self.validation_folder:
                print('Previously saved validation plots have been overwritten')
        else:
            os.mkdir(self.path + folder)
