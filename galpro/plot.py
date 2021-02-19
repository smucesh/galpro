from sys import exit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from .utils import *
from .conf import set_plot_params


class Plot:
    """
    Internal class for creating various plots.

    Parameters
    ----------
    y_test: array_like
        An array of target features of testing galaxies with the same shape as y_train.

    target_features: list
        A list of variables of target features.

    path: str
        Location of the model directory.
    """

    def __init__(self, y_test, y_pred, posteriors, validation, target_features, no_samples, no_features, path):

        # Initialise arguments
        self.y_test = y_test
        self.y_pred = y_pred
        self.posteriors = posteriors
        self.validation = validation
        self.target_features = target_features
        self.no_samples = no_samples
        self.no_features = no_features
        self.path = path

        # Internal parameters
        self.point_estimate_folder = 'point_estimates/'
        self.posterior_folder = 'posteriors/'
        self.validation_folder = 'validation/'
        self.plot_folder = 'plots/'

        # Initialise plotting aesthetics
        self.no_bins, self.no_points = set_plot_params()

    def plot_scatter(self, show=False, save=True):
        """Creates scatter plots"""

        # Check if y_test is available
        if self.y_test is None:
            print('Pass in y_test to generate scatter plots.')
            exit()

        # Load point estimates
        if self.y_pred is None:
            self.y_pred = load_point_estimates(path=self.path)

        # Convert if 1d array
        if self.no_features == 1:
            self.y_pred, = convert_1d_arrays(self.y_pred)

        # Get metrics
        metrics = get_pred_metrics(y_test=self.y_test, y_pred=self.y_pred, no_features=self.no_features)

        for feature in np.arange(self.no_features):
            min_, max_ = [np.floor(np.min(self.y_test[:, feature])), np.ceil(np.max(self.y_test[:, feature]))]
            sns.scatterplot(x=self.y_test[:, feature], y=self.y_pred[:, feature], color='purple', edgecolor='purple',
                            alpha=0.6, marker='.')
            plt.plot([min_, max_], [min_, max_], color='black', linestyle='--', linewidth='1')
            plt.plot([], [], ' ', label=f'$\sigma_{{NMAD}}: {metrics[feature]:.3f}$')
            plt.xlim([min_, max_])
            plt.ylim([min_, max_])
            plt.xlabel(self.target_features[feature])
            plt.ylabel(self.target_features[feature] + '$_{ML}$')
            plt.legend(edgecolor='None', loc='lower right', framealpha=0)

            if save:
                plt.savefig(self.path + self.point_estimate_folder + self.plot_folder + 'var_' + str(feature)
                            + '_scatter.png', bbox_inches='tight')
            if show:
                plt.show()

            plt.close()

    def plot_marginal(self, show=False, save=True):
        """Creates marginal PDF plots"""

        # Check if 1d
        if self.no_features != 1:
            print('Number of target features is not equal to 1. Please use plot_joint() or plot_corner().')
            exit()

        # Load point estimates
        if self.y_pred is None:
            self.y_pred = load_point_estimates(path=self.path)

        # Load posteriors
        if self.posteriors is None:
            self.posteriors = load_posteriors(path=self.path)

        for sample in np.arange(self.no_samples):
            posterior = self.posteriors[str(sample)][:].reshape(-1,)
            sns.kdeplot(posterior, color="darkorchid", shade=True)
            plt.axvline(self.y_pred[sample], color='black', linestyle='--', linewidth='1', label='$Predicted$')

            if self.y_test is not None:
                plt.axvline(self.y_test[sample], color='gold', linestyle='--', linewidth='1', label='$True$')
                plt.legend(framealpha=0, edgecolor='None', loc='upper left')
            else:
                plt.legend(framealpha=0, edgecolor='None', loc='upper left')

            plt.xlabel(self.target_features[0])
            plt.ylabel(f'$f(${self.target_features[0]}$)$')

            if save:
                plt.savefig(self.path + self.posterior_folder + self.plot_folder + 'marginal_pdf_' + str(sample) +
                            '.png', bbox_inches='tight')
            if show:
                plt.show()

            plt.close()

    def plot_joint(self, show=False, save=True):
        """Creates joint PDF plots"""

        if self.no_features == 1 or self.no_features > 2:
            print('Number of target features is less than or greater than 2. Please use plot_marginal or plot_corner().'
                  '')
            exit()

        # Load point estimates
        if self.y_pred is None:
            self.y_pred = load_point_estimates(path=self.path)

        # Load posteriors
        if self.posteriors is None:
            self.posteriors = load_posteriors(path=self.path)

        for sample in np.arange(self.no_samples):
            posterior = self.posteriors[str(sample)][:]
            g = sns.jointplot(x=posterior[:, 0], y=posterior[:, 1], kind="kde", space=0.1, color="darkorchid",
                              n_levels=10, ratio=4,
                              marginal_kws={'lw': 3, 'color': 'darkorchid', 'shade': True, 'alpha': 0.6})
            g.plot_joint(plt.scatter, color="green", s=15, marker="o", alpha=0.6, edgecolor='black')
            g.ax_joint.collections[0].set_alpha(0)
            g.set_axis_labels(self.target_features[0], self.target_features[1])

            g.ax_marg_x.axvline(self.y_pred[sample, 0], color='white', linestyle='--', linewidth='2')
            g.ax_marg_y.axhline(self.y_pred[sample, 1], color='white', linestyle='--', linewidth='2')
            predicted, = plt.plot(self.y_pred[sample, 0], self.y_pred[sample, 1], color='white', marker='*',
                                  markersize=10, linestyle='None', label='$Predicted$')

            if self.y_test is not None:
                g.ax_marg_x.axvline(self.y_test[sample, 0], color='gold', linestyle='--', linewidth='2')
                g.ax_marg_y.axhline(self.y_test[sample, 1], color='gold', linestyle='--', linewidth='2')
                true, = plt.plot(self.y_test[sample, 0], self.y_test[sample, 1], color='gold', marker='*',
                                 markersize=10, linestyle='None', label='$True$')
                plt.legend(handles=[true, predicted], facecolor='lightgrey', loc='lower right')
            else:
                plt.legend(handles=[predicted], facecolor='lightgrey', loc='lower right')

            sns.despine(top=False, left=False, right=False, bottom=False)

            if save:
                plt.savefig(self.path + self.posterior_folder + self.plot_folder + 'joint_pdf_' + str(sample) + '.png',
                            bbox_inches='tight')
            if show:
                plt.show()

            plt.close()

    def plot_corner(self, show=False, save=True):
        """Creates corner PDF plots"""

        if self.no_features < 2:
            print('Number of target features is 1. Please use plot_marginal().')
            exit()

        # Load point estimates
        if self.y_pred is None:
            self.y_pred = load_point_estimates(path=self.path)

        # Load posteriors
        if self.posteriors is None:
            self.posteriors = load_posteriors(path=self.path)

        # Get quantiles
        quantiles = get_quantiles(posteriors=self.posteriors, no_samples=self.no_samples, no_features=self.no_features)

        for sample in np.arange(self.no_samples):
            posterior = self.posteriors[str(sample)][:]
            posterior = pd.DataFrame(posterior, columns=self.target_features)
            #posterior = pd.DataFrame(np.array(posteriors[sample]), columns=self.target_features)
            g = sns.PairGrid(data=posterior, corner=True)
            g = g.map_lower(sns.kdeplot, shade=True, color='darkorchid', n_levels=10, shade_lowest=False)
            g = g.map_diag(sns.kdeplot, lw=2, color='darkorchid', shade=True)

            for feature in np.arange(self.no_features):
                g.axes[feature, feature].set_title(f'{self.target_features[feature]}'
                                                   f'$= {quantiles[feature, sample, 1]:.2f}'
                                                   f'^{{+{quantiles[feature, sample, 2]-quantiles[feature, sample, 1]:.2f}}}'
                                                   f'_{{-{quantiles[feature, sample, 1]-quantiles[feature, sample, 0]:.2f}}}$',
                                                   fontsize=10)
                g.axes[feature, feature].axvline(quantiles[feature, sample, 0], color='black', linestyle='--',
                                                 linewidth=1)
                g.axes[feature, feature].axvline(quantiles[feature, sample, 1], color='black', linestyle='--',
                                                 linewidth=1)
                g.axes[feature, feature].axvline(quantiles[feature, sample, 2], color='black', linestyle='--',
                                                 linewidth=1)

            sns.despine(top=False, left=False, right=False, bottom=False)

            if save:
                plt.savefig(self.path + self.posterior_folder + self.plot_folder + 'corner_pdf_' + str(sample) + '.png'
                            , bbox_inches='tight')
            if show:
                plt.show()

            plt.close()

    def plot_pit(self, show=False, save=True):
        """Creates probability integral transform (PIT) distribution plots"""

        # Load PITs
        if self.validation is None:
            self.validation = load_validation(path=self.path)

        pits = self.validation['pits'][:]

        # Get marginal pdf metrics
        outliers, kld, kst, cvm = get_pdf_metrics(pits=pits, no_samples=self.no_samples, no_features=self.no_features,
                                                  no_bins=self.no_bins, coppits=None)

        for feature in np.arange(self.no_features):
            qqplot = sm.qqplot(pits[:, feature], 'uniform', line='45').gca().lines
            qq_theory, qq_data = [qqplot[0].get_xdata(), qqplot[0].get_ydata()]
            plt.close()

            ax1 = sns.distplot(pits[:, feature], bins=self.no_bins, kde=False,
                               hist_kws={'histtype': 'stepfilled', 'color': 'slategrey', 'edgecolor': 'slategrey',
                                         'alpha': 0.5})
            ax2 = plt.twinx()
            ax2 = sns.lineplot(x=qq_theory, y=qq_data, color='blue')
            ax2.plot([0, 1], [0, 1], color='black', linewidth=1, linestyle='--')
            plt.plot([], [], ' ', label=f'$Outliers: {outliers[feature]:.2f}\%$')
            plt.plot([], [], ' ', label=f'$KLD: {kld[feature]:.3f}$')
            plt.plot([], [], ' ', label=f'$KST: {kst[feature]:.3f}$')
            plt.plot([], [], ' ', label=f'$CvM: {cvm[feature]:.3f}$')

            ax1.set_xlabel('$Q_{theory} \ | \ PIT \ ($' + self.target_features[feature] + '$)$')
            ax1.set_ylabel('$N$')
            ax2.set_ylabel('$Q_{data}$')
            ax2.set_xlim([0, 1])
            ax2.set_ylim([0, 1])

            leg = plt.legend(framealpha=0, edgecolor='None', loc='lower right')
            hp = leg._legend_box.get_children()[1]
            for vp in hp.get_children():
                for row in vp.get_children():
                    row.set_width(100)
                    row.mode = "expand"
                    row.align = "right"

            if save:
                plt.savefig(self.path + self.validation_folder + self.plot_folder + 'var_' + str(feature) + '_pit.png',
                            bbox_inches='tight')
            if show:
                plt.show()

            plt.close()

    def plot_coppit(self, show=False, save=True):
        """Creates copula probability integral transform (copPIT) distribution plots"""

        if self.no_features < 2:
            print('Cannot plot copPIT as number of target features is less than 2.')
            exit()

        # Load copPITs
        if self.validation is None:
            self.validation = load_validation(path=self.path)

        pits = self.validation['pits'][:]
        coppits = self.validation['coppits'][:]

        # Get full pdf metrics
        outliers, kld, kst, cvm = get_pdf_metrics(pits=pits, no_samples=self.no_samples, no_features=self.no_features,
                                                  no_bins=self.no_bins, coppits=coppits)

        qqplot = sm.qqplot(coppits, 'uniform', line='45').gca().lines
        qq_theory, qq_data = [qqplot[0].get_xdata(), qqplot[0].get_ydata()]
        plt.close()

        ax1 = sns.distplot(coppits, bins=self.no_bins, kde=False,
                           hist_kws={'histtype': 'stepfilled', 'color': 'slategrey', 'edgecolor': 'slategrey',
                                     'alpha': 0.5}
                           )
        ax2 = plt.twinx()
        ax2 = sns.lineplot(x=qq_theory, y=qq_data, color='blue')
        ax2.plot([0, 1], [0, 1], color='black', linewidth=1, linestyle='--')
        plt.plot([], [], ' ', label=f'$Outliers: {outliers:.2f}\%$')
        plt.plot([], [], ' ', label=f'$KLD: {kld:.3f}$')
        plt.plot([], [], ' ', label=f'$KST: {kst:.3f}$')
        plt.plot([], [], ' ', label=f'$CvM: {cvm:.3f}$')

        ax1.set_xlabel('$Q_{theory} \ | \ copPIT$')
        ax1.set_ylabel('$N$')
        ax2.set_ylabel('$Q_{data}$')
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])

        leg = plt.legend(framealpha=0, edgecolor='None', loc='lower right')
        hp = leg._legend_box.get_children()[1]
        for vp in hp.get_children():
            for row in vp.get_children():
                row.set_width(100)
                row.mode = "expand"
                row.align = "right"

        if save:
            plt.savefig(self.path + self.validation_folder + self.plot_folder + 'coppit.png', bbox_inches='tight')
        if show:
            plt.show()

        plt.close()

    def plot_marginal_calibration(self, show=False, save=True):
        """Creates marginal calibration plots"""

        # Load marginal calibration
        if self.validation is None:
            self.validation = load_validation(path=self.path)

        marginal_calibration = self.validation['marginal_calibration'][:]

        for feature in np.arange(self.no_features):
            sns.lineplot(x=marginal_calibration[0, :, feature], y=marginal_calibration[1, :, feature], color="blue")
            plt.axhline(0, color='black', linewidth=1, linestyle='--')
            plt.ylim([-np.max(np.abs(marginal_calibration[1, :, feature])),
                      np.max(np.abs(marginal_calibration[1, :, feature]))])
            plt.xlabel(self.target_features[feature])
            plt.ylabel(r'$\hat{F}_{I} - \tilde{G}_{I}$')

            if save:
                plt.savefig(self.path + self.validation_folder + self.plot_folder + 'var_' + str(feature) +
                            '_marginal_calibration.png', bbox_inches='tight')
            if show:
                plt.show()

            plt.close()

    def plot_kendall_calibration(self, show=False, save=True):
        """Creates kendall calibration plots"""

        if self.no_features < 2:
            print('Cannot plot kendall calibration as number of target features is less than 2.')
            exit()

        # Load kendall calibration
        if self.validation is None:
            self.validation = load_validation(path=self.path)

        kendall_calibration = self.validation['kendall_calibration'][:]

        sns.lineplot(x=np.linspace(0, 1, self.no_points), y=kendall_calibration, color="blue")
        plt.axhline(0, color='black', linewidth=1, linestyle='--')
        plt.xlim([0, 1])
        plt.ylim([-np.max(np.abs(kendall_calibration)), np.max(np.abs(kendall_calibration))])
        plt.xlabel('$w$')
        plt.ylabel(r'$\mathcal{\hat{K}}_{H_{I}}  - \tilde{J}_{I}$')

        if save:
            plt.savefig(self.path + self.validation_folder + self.plot_folder + 'kendall_calibration.png',
                        bbox_inches='tight')
        if show:
            plt.show()

        plt.close()