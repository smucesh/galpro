import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm

from galpro.utils import *
from galpro.conf import set_plot_params


class Plot:

    def __init__(self, y_test, target_features, path):

        # Initialise arguments
        self.y_test = y_test
        self.target_features = target_features
        self.path = path
        self.no_features = len(target_features)
        self.no_points = 100
        self.point_estimate_folder = 'point_estimates/'
        self.posterior_folder = 'posteriors/'
        self.validation_folder = 'validation/'

        if self.y_test is not None:
            self.no_samples = self.y_test.shape[0]

        # Initialise plotting aesthetics
        set_plot_params()

    def plot_scatter(self):

        # Check if y_test is available
        if self.y_test is None:
            print('Pass in y_test to generate scatter plots.')
            exit()

        # Load point estimates
        y_pred = load_point_estimates(path=self.path)

        if self.no_features == 1:
            y_pred = convert_1d_arrays(y_pred)

        # Get metrics
        metrics = get_pred_metrics(y_test=self.y_test, y_pred=y_pred)

        for feature in np.arange(self.no_features):
            min_, max_ = [np.floor(np.min(self.y_test[:, feature])), np.ceil(np.max(self.y_test[:, feature]))]
            sns.scatterplot(x=self.y_test[:, feature], y=y_pred[:, feature], color='purple', edgecolor='purple',
                            alpha=0.6, marker='.')
            plt.plot([min_, max_], [min_, max_], color='black', linestyle='--', linewidth='1')
            plt.plot([], [], ' ', label=f'$RMSE: {metrics[feature]}$')
            plt.xlim([min_, max_])
            plt.ylim([min_, max_])
            plt.xlabel('$' + self.target_features[feature] + '$')
            plt.ylabel('$' + self.target_features[feature] + '_{ML}$')
            plt.legend(edgecolor='None', loc='lower right', framealpha=0)
            plt.savefig(self.path + self.point_estimate_folder + 'plots/' + str(feature) + '_scatter.png',
                        bbox_inches='tight', dpi=300)
            plt.close()

        print('Scatter plots have been created.')

    def plot_marginal(self):

        # Load point estimates
        y_pred = load_point_estimates(path=self.path)

        # Load posteriors
        pdfs = load_posteriors(path=self.path)

        for sample in np.arange(self.no_samples):
            pdf = np.array(pdfs[sample]).reshape(-1,)
            sns.kdeplot(pdf, color="darkorchid", shade=True)
            plt.axvline(y_pred[sample], color='black', linestyle='--', linewidth='1', label='$Predicted$')

            if self.y_test is not None:
                plt.axvline(self.y_test[sample], color='gold', linestyle='--', linewidth='1', label='$True$')
                plt.legend(framealpha=0, edgecolor='None', loc='upper left')
            else:
                plt.legend(framealpha=0, edgecolor='None', loc='upper left')

            plt.xlabel('$' + self.target_features[0] + '$')
            plt.ylabel('$N$')
            plt.savefig(self.path + self.posterior_folder + 'plots/' + 'marginal_pdf_' + str(sample) + '.png',
                        bbox_inches='tight', dpi=300)
            plt.close()

        print('Marginal plots have been created.')

    def plot_posterior(self):

        # Load point estimates
        y_pred = load_point_estimates(path=self.path)

        # Load posteriors
        pdfs = load_posteriors(path=self.path)

        for sample in np.arange(self.no_samples):
            pdf = np.array(pdfs[sample])
            g = sns.jointplot(x=pdf[:, 0], y=pdf[:, 1], kind="kde", space=0.1, color="darkorchid", n_levels=10, ratio=4,
                              marginal_kws={'lw': 3, 'color': 'darkorchid', 'shade': True, 'alpha': 0.6})
            g.plot_joint(plt.scatter, color="green", s=15, marker="o", alpha=0.6, edgecolor='black')
            g.ax_joint.collections[0].set_alpha(0)
            g.set_axis_labels(self.target_features[0], self.target_features[1])

            g.ax_marg_x.axvline(y_pred[sample, 0], color='white', linestyle='--', linewidth='2')
            g.ax_marg_y.axhline(y_pred[sample, 1], color='white', linestyle='--', linewidth='2')
            predicted, = plt.plot(y_pred[sample, 0], y_pred[sample, 1], color='white', marker='*', markersize=10,
                                  linestyle='None', label='$Predicted$')

            if self.y_test is not None:
                g.ax_marg_x.axvline(self.y_test[sample, 0], color='gold', linestyle='--', linewidth='2')
                g.ax_marg_y.axhline(self.y_test[sample, 1], color='gold', linestyle='--', linewidth='2')
                true, = plt.plot(self.y_test[sample, 0], self.y_test[sample, 1], color='gold', marker='*', markersize=10,
                                 linestyle='None', label='$True$')
                plt.legend(handles=[true, predicted], facecolor='lightgrey', loc='lower right')
            else:
                plt.legend(handles=[predicted], facecolor='lightgrey', loc='lower right')

            sns.despine(top=False, left=False, right=False, bottom=False)
            plt.savefig(self.path + self.posterior_folder + 'plots/' + 'joint_pdf_' + str(sample) + '.png',
                        bbox_inches='tight', dpi=300)
            plt.close()

        print('Posterior plots have been created.')

    def plot_corner(self):

        # Load point estimates
        y_pred = load_point_estimates(path=self.path)

        # Load posteriors
        pdfs = load_posteriors(path=self.path)

        # Get quantiles
        quantiles = get_quantiles(pdfs=pdfs)

        for sample in np.arange(self.no_samples):
            pdf = pd.DataFrame(np.array(pdfs[sample]), columns=self.target_features)
            g = sns.PairGrid(data=pdf, corner=True)
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
            plt.savefig(self.path + self.posterior_folder + 'plots/' + 'corner_plot_' + str(sample) + '.png',
                        bbox_inches='tight', dpi=300)
            plt.close()

        print('Corner plots have been created.')

    def plot_pit(self):

        # Load PITs
        pit = np.load(self.path + self.validation_folder + 'pits.npy')

        # Get marginal pdf metrics
        outliers, kld, kst, cvm = get_pdf_metrics(data=pit, no_features=self.no_features)

        for feature in np.arange(self.no_features):
            qqplot = sm.qqplot(pit[:, feature], 'uniform', line='45').gca().lines
            qq_theory, qq_data = [qqplot[0].get_xdata(), qqplot[0].get_ydata()]
            plt.close()

            ax1 = sns.distplot(pit[:, feature], bins=self.no_points, kde=False,
                               hist_kws={'histtype': 'stepfilled', 'color': 'slategrey', 'edgecolor': 'slategrey',
                                         'alpha': 0.5})
            ax2 = plt.twinx()
            ax2 = sns.lineplot(x=qq_theory, y=qq_data, color='blue')
            ax2.plot([0, 1], [0, 1], color='black', linewidth=1, linestyle='--')
            plt.plot([], [], ' ', label=f'$Outliers: {outliers[feature]:.2f}\%$')
            plt.plot([], [], ' ', label=f'$KLD: {kld[feature]:.3f}$')
            plt.plot([], [], ' ', label=f'$KST: {kst[feature]:.3f}$')
            plt.plot([], [], ' ', label=f'$CvM: {cvm[feature]:.3f}$')

            ax1.set_xlabel('$Q_{theory}/PIT$')
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

            plt.savefig(self.path + self.validation_folder + 'plots/' + str(feature) + '_pit.png',
                        bbox_inches='tight', dpi=300)
            plt.close()

        print('PIT plots have been created.')

    def plot_coppit(self):

        # Load copPITs
        coppit = np.load(self.path + self.validation_folder + 'coppits.npy')

        # Get full pdf metrics
        outliers, kld, kst, cvm = get_pdf_metrics(data=coppit, no_features=1)

        qqplot = sm.qqplot(coppit, 'uniform', line='45').gca().lines
        qq_theory, qq_data = [qqplot[0].get_xdata(), qqplot[0].get_ydata()]
        plt.close()

        ax1 = sns.distplot(coppit, bins=self.no_points, kde=False,
                           hist_kws={'histtype': 'stepfilled', 'color': 'slategrey', 'edgecolor': 'slategrey',
                                     'alpha': 0.5}
                           )
        ax2 = plt.twinx()
        ax2 = sns.lineplot(x=qq_theory, y=qq_data, color='blue')
        ax2.plot([0, 1], [0, 1], color='black', linewidth=1, linestyle='--')
        plt.plot([], [], ' ', label=f'$Outliers: {outliers[0]:.2f}\%$')
        plt.plot([], [], ' ', label=f'$KLD: {kld[0]:.3f}$')
        plt.plot([], [], ' ', label=f'$KST: {kst[0]:.3f}$')
        plt.plot([], [], ' ', label=f'$CvM: {cvm[0]:.3f}$')

        ax1.set_xlabel('$Q_{theory}/copPIT$')
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

        plt.savefig(self.path + self.validation_folder + 'plots/' + 'coppit.png', bbox_inches='tight', dpi=300)
        plt.close()

        print('copPIT plot has been created.')

    def plot_marginal_calibration(self):

        # Load marginal calibration
        marginal_calibration = np.load(self.path + self.validation_folder + 'marginal_calibration.npy')

        for feature in np.arange(self.no_features):
            min_, max_ = [np.floor(np.min(self.y_test[:, feature])), np.ceil(np.max(self.y_test[:, feature]))]
            sns.lineplot(x=np.linspace(min_, max_, self.no_points), y=marginal_calibration[:, feature], color="blue")
            plt.axhline(0, color='black', linewidth=1, linestyle='--')
            plt.ylim([-np.max(marginal_calibration), np.max(marginal_calibration)])
            plt.xlabel(self.target_features[feature])
            plt.ylabel(r'$\hat{F}_{I} - \tilde{G}_{I}$')
            plt.savefig(self.path + self.validation_folder + 'plots/' + str(feature) + '_marginal_calibration.png',
                        bbox_inches='tight', dpi=300)
            plt.close()

        print('Marginal calibration plots have been created')

    def plot_kendall_calibration(self):

        # Load kendall calibration
        kendall_calibration = np.load(self.path + self.validation_folder + 'kendall_calibration.npy')

        sns.lineplot(x=np.linspace(0, 1, self.no_points), y=kendall_calibration, color="blue")
        plt.axhline(0, color='black', linewidth=1, linestyle='--')
        plt.xlim([0, 1])
        plt.ylim([-np.max(kendall_calibration), np.max(kendall_calibration)])
        plt.xlabel('$w$')
        plt.ylabel(r'$\mathcal{\hat{K}}_{H_{I}}  - \tilde{J}_{I}$')
        plt.savefig(self.path + self.validation_folder + 'plots/' + 'kendall_calibration.png',
                    bbox_inches='tight', dpi=300)
        plt.close()

        print('Kendall calibration plot has been created.')