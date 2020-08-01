import matplotlib.patches as mpatches
# from tsnecuda import TSNE
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from operator import itemgetter

fontP = FontProperties()
fontP.set_size('small')

OUT_DIR = 'vis_outputs/'
PAMAP_COLOR_DICT = dict({1: 'black', 2: 'red', 3: 'gold', 4: 'deepskyblue', 5: 'grey',
                         6: 'olive', 7: 'indigo', 12: 'deeppink', 13: 'orange',
                         16: 'lightblue', 17: 'teal', 24: 'brown'})
DSADS_COLOR_DICT = dict({1: 'black', 2: 'red', 3: 'gold', 4: 'deepskyblue', 5: 'grey',
                         6: 'olive', 7: 'indigo', 8: 'deeppink', 9: 'orange',
                         10: 'lightblue', 11: 'teal', 12: 'brown', 13: 'lime', 14: 'mediumblue',
                         15: 'mediumspringgreen', 16: 'lightsalmon', 17: 'lightsteelblue',
                         18: 'orchid', 19: 'sandybrown'})


class Visualizer():
    def __init__(self, reversed_virtual_map, reversed_original_map, args, total_batches, tp, holdout_size):
        self.args = args
        self.tp = tp
        self.holdout_size = holdout_size
        self.previous_class_scores = {}
        self.current_class_scores = {}
        self.batch_num = total_batches
        self.incr_state_to_scores = {i: () for i in range(1, self.batch_num)}
        self.incr_step_to_norm_params = {i: {'old_wts': [], 'old_bias': [], 'new_wts': [], 'new_bias': []} for i
                                         in range(1, self.batch_num)}
        self.epoch_to_norm = {i: {} for i in range(self.args.epochs)}
        self.detailed_micro_scores = [() for i in range(self.batch_num - 1)]
        self.detailed_macro_scores = [() for i in range(self.batch_num - 1)]
        self.virtual_map = reversed_virtual_map
        self.original_map = reversed_original_map

    def plot_tsne(self, data, labels, itera=0, test=False):
        # out_path = OUT_DIR + 'tsne_vis/'
        # color_coding = PAMAP_COLOR_DICT if self.args.dataset == 'pamap' else DSADS_COLOR_DICT
        # X_embed = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(
        #     data)
        # if itera > 0:
        #     sns_plot = sns.scatterplot(X_embed[:, 0], X_embed[:, 1],
        #                                hue=[int(i) for i in list(itemgetter(*list(itemgetter(*labels)(self.virtual_map)))(
        #                                    self.original_map))],
        #                                legend='full', palette=sns.color_palette("hls", len(set(labels))))
        # else:
        #     sns_plot = sns.scatterplot(X_embed[:, 0], X_embed[:, 1],
        #                                hue=[int(i) for i in
        #                                     list(itemgetter(*labels)(self.original_map))],
        #                                legend='full', palette=sns.color_palette("hls", len(set(labels))))
        # fig = sns_plot.get_figure()
        # plt.tick_params(axis='both', which='major', labelsize=11)
        #
        # if test:
        #     box = sns_plot.get_position()
        #     sns_plot.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
        #     lgd = sns_plot.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', borderaxespad=0., title='Class ID',
        #                     fontsize=11, fancybox=True, shadow=True)
        #     fig.savefig(
        #         out_path + f'{self.args.dataset}_iter_{itera}_tp_{self.tp}_{self.args.method}_test.pdf',
        #         dpi=800, bbox_extra_artists=(lgd,), bbox_inches='tight')
        # else:
        #     sns_plot.legend_.remove()
        #     fig.savefig(
        #         out_path + f'{self.args.dataset}_iter_{itera}_tp_{self.tp}_{self.args.method}_train.pdf',
        #         dpi=800, bbox_inches='tight')
        # plt.show()
        # fig.clf()
        pass

    def plot_weight_norms_by_epochs(self, itera, seen_cls):
        out_path = OUT_DIR + 'norm_vis/by_epochs/'
        epochs = list(self.epoch_to_norm.keys())
        colors = ['green', 'red', 'black', 'blue', 'orange', 'saddlebrown']
        patchlist = []
        SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        ax = plt.subplot(111)
        # ax.xaxis.tick_top()
        print(seen_cls, epochs,  self.epoch_to_norm[0].keys(), self.epoch_to_norm[1].keys())
        for label in range(seen_cls):
            print(label)
            norm_over_epochs = [self.epoch_to_norm[each][label] for each in epochs]
            datakey = mpatches.Patch(color=colors[label], label='||w{}||'.format(label).translate(SUB))
            patchlist.append(datakey)
            ax.plot(epochs, norm_over_epochs, color=colors[label], linewidth=2.0)
        batches = [i for i in range(0, len(epochs) + len(epochs) // 5, len(epochs)//5)]
        ax.set_xticks(batches)
        ax.set_xticklabels(batches)
        for x_label in batches:
            ax.axvline(x=x_label, linestyle='--', color='aliceblue',
                    zorder=0)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        ax.legend(handles=patchlist, fontsize=11, bbox_to_anchor=(1, 0.5), loc='center left',
                   fancybox=True, shadow=True)
        # ax.xaxis.set_label_position('top')
        plt.xlabel('Training epochs')
        plt.ylabel('L2 norm of weights')
        plt.tick_params(axis='both', which='major', labelsize=11)
        plt.savefig(
            out_path + f'{self.args.dataset}_iter_{itera}_tp_{self.tp}_size_{self.holdout_size}_{self.args.method}.png', dpi=600)
        plt.show()
        plt.clf()

    def visualize_new_and_old_weights(self, weights, itera, seen_cls, classes_by_groups, biases=None):
        out_path = OUT_DIR + 'norm_vis/by_batches/'
        for key in self.incr_step_to_norm_params[itera].keys():
            if key == 'old_wts':
                old_wts = weights[:seen_cls - self.args.new_classes]
                old_wts = [torch.norm(item, p=2, dim=0).item() for item in old_wts]
                self.incr_step_to_norm_params[itera][key] = old_wts
            elif key == 'old_bias' and biases is not None:
                old_biases = biases[:seen_cls - self.args.new_classes]
                old_biases = [torch.norm(item, p=2, dim=0).item() for item in old_biases]
                self.incr_step_to_norm_params[itera][key] = old_biases
            elif key == 'new_wts':
                new_wts = weights[seen_cls - self.args.new_classes:seen_cls]
                new_wts = [torch.norm(item, p=2, dim=0).item() for item in new_wts]
                self.incr_step_to_norm_params[itera][key] = new_wts
            elif key == 'new_bias' and biases is not None:
                new_biases = weights[seen_cls - self.args.new_classes:seen_cls]
                new_biases = [torch.norm(item, p=2, dim=0).item() for item in new_biases]
                self.incr_step_to_norm_params[itera][key] = new_biases

        if itera == self.batch_num - 1:
            n_rows = 2 if biases is not None else 1
            f, axes = plt.subplots(nrows=n_rows, ncols=self.batch_num - 1, sharey='row')
            class_intervals = [len([item for each in classes_by_groups[:iter + 1] for item in each]) for iter in
                               range(itera + 1)]
            for col in range(1, itera + 1):
                x_labels_base = [idx + 0.2 for idx in np.linspace(0.2, class_intervals[0],
                                                                  self.args.base_classes,
                                                                  endpoint=False)]
                x_labels_old = [idx + 0.2 for idx in np.linspace(class_intervals[0] + 0.2, class_intervals[col - 1],
                                                                 len(self.incr_step_to_norm_params[col][
                                                                         'old_wts']) - self.args.base_classes,
                                                                 endpoint=False)]
                x_labels_new = [idx + 0.2 for idx in np.linspace(class_intervals[col - 1] + 0.2, class_intervals[col],
                                                                 len(self.incr_step_to_norm_params[col]['new_wts']),
                                                                 endpoint=False)]
                row_index = axes[0] if n_rows > 1 else axes  # take the first row for plotting weights if n_rows > 1
                bc = row_index[col - 1].scatter(x_labels_base,
                                                self.incr_step_to_norm_params[col]['old_wts'][:self.args.base_classes],
                                                s=25, c='blue')
                if col > 1:
                    oc = row_index[col - 1].scatter(x_labels_old, self.incr_step_to_norm_params[col]['old_wts'][
                                                                  self.args.base_classes:], s=25, c='red')
                nc = row_index[col - 1].scatter(x_labels_new, self.incr_step_to_norm_params[col]['new_wts'], s=25,
                                                c='green')
                plt.sca(row_index[col - 1])
                plt.xticks(class_intervals[:col + 1])
                for xc in class_intervals[:col]:
                    plt.axvline(x=xc, color='grey', linestyle='--', linewidth=0.9)
                if col == 1:
                    plt.ylabel('L2 norm of weights', fontsize=11)
                if biases is not None:
                    # biases are None if CosineLinearLayer() used. i.e., args.method contains 'cn'
                    axes[1][col - 1].scatter(x_labels_base,
                                             self.incr_step_to_norm_params[col]['old_bias'][:self.args.base_classes],
                                             s=25, c='blue')
                    if col > 1:
                        axes[1][col - 1].scatter(x_labels_old, self.incr_step_to_norm_params[col]['old_bias'][
                                                               self.args.base_classes:], s=25, c='red')
                    axes[1][col - 1].scatter(x_labels_new, self.incr_step_to_norm_params[col]['new_bias'], s=25,
                                             c='green')
                    plt.sca(axes[1][col - 1])
                    plt.xticks(class_intervals[:col + 1])
                    for xc in class_intervals[:col]:
                        plt.axvline(x=xc, color='grey', linestyle='--', linewidth=1.0)
                    if col == 1:
                        plt.ylabel('L2 norm of biases', fontsize=11)
            if col == itera:
                plt.legend((bc, oc, nc), ('base classes', 'old classes', 'new classes'), fontsize=11)
            f.text(0.5, 0.02, 'No. of observed classes', va='center', ha='center', fontsize=11)
            f.tight_layout()
            plt.tick_params(axis='both', which='major', labelsize=11)
            f.savefig(
                out_path + f'{self.args.dataset}_iter_{itera}_tp_{self.tp}_size_{self.holdout_size}_{self.args.method}.png',
                dpi=600)
            plt.show(f)
            plt.close(f)

    def plot_detailed_accuracies(self):
        out_path = OUT_DIR + 'accuracy_vis/detail_acc/'
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        base_scores = [each[0] for each in self.detailed_micro_scores]
        old_scores = [each[1] for each in self.detailed_micro_scores]
        new_scores = [each[2] for each in self.detailed_micro_scores]
        all_scores = [each[3] for each in self.detailed_micro_scores]
        batches = [i for i in range(1, self.batch_num)]

        for score in [base_scores, old_scores, new_scores, all_scores]:
            print(score)
            ax1.plot(batches, score)
        ax1.set_xticks(batches)
        ax1.set_xticklabels(batches)
        ax1.legend(labels=['Base classes', 'Old classes', 'New classes', 'All classes'])
        ax1.set_title("Detailed accuracy comparison")
        fig.savefig(out_path + f'{self.args.dataset}_tp_{self.tp}_size_{self.holdout_size}_{self.args.method}.png')
        # plt.show()
        plt.close(fig)

    def plot_by_states(self):
        out_path = OUT_DIR + 'accuracy_vis/acc_by_batches/'
        print(f" state to scores: {self.incr_state_to_scores}")

        ncols = 2 if 'bic' in self.args.method or 'wa' in self.args.method else 1
        fig, ax = plt.subplots(nrows=1, ncols=ncols, sharex=True, sharey=True)

        print(self.incr_state_to_scores)
        for x, y in self.incr_state_to_scores.items():
            for col in range(ncols):
                axis_ref = ax[col] if ncols > 1 else ax
                axis_ref.scatter(x, y[col * 2], marker="D", c='orangered', s=25)  # o for previous class scores
                axis_ref.scatter(x, y[col * 2 + 1], marker="s", c='green', s=30)  # x for current class scores
                box = axis_ref.get_position()
                axis_ref.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        states = [i for i in range(self.batch_num)]
        axis_ref = ax if ncols == 1 else ax[0]

        for col in range(ncols):
            ax[col].set_xticks(np.arange(len(states)))
            ax[col].set_xticklabels(states)
            if ncols > 0 and col == 1:
                ax[col].set_title("After bias removal", fontsize=11)
                ax[col].legend(labels=['Old classes', 'New classes'], fontsize=11, bbox_to_anchor=(1, 0.5), loc='center left')


        axis_ref.set_ylabel('Average accuracy', fontsize=11)
        axis_ref.set_title("Before bias removal", fontsize=11)
        for x, y in self.incr_state_to_scores.items():
            for col in range(ncols):
                axis_ref = ax[col] if ncols > 1 else ax
                axis_ref.axvline(x=x, linestyle='--', color='aliceblue',
                                 zorder=0)
        plt.tick_params(axis='both', which='major', labelsize=11)
        fig.text(0.5, 0.03, 'No. of incremental tasks', ha='center', va='center', fontsize=11)
        # fig.tight_layout()
        fig.savefig(out_path + f'{self.args.dataset}_tp_{self.tp}_size_{self.holdout_size}_{self.args.method}.png')
        plt.show()
        plt.close(fig)

    def plot_by_classes(self):
        out_path = OUT_DIR + 'accuracy_vis/acc_by_classes/'
        common_classes = list(set(self.current_class_scores).intersection(self.previous_class_scores))
        mean_of_previous_classes = {key: sum(val) / len(val) for key, val in self.previous_class_scores.items()}
        print(f"Common classes: {common_classes}\nMean of prev classes: {mean_of_previous_classes}")

        if (len(common_classes) > 0):
            for x, y in mean_of_previous_classes.items():
                plt.scatter(x, y, marker="D", c='r')  # o for previous class scores
                plt.scatter(x, self.current_class_scores[x], marker="s", c='b')  # x for current class scores
            plt.xticks(common_classes)
            plt.axes().set_xticklabels(map(int, common_classes))
            plt.title("Scores of classes when they first appear vs average of old scores.")
            plt.legend(labels=['average of old scores', 'score when new'])
            plt.savefig(out_path + f'{self.args.dataset}_tp_{self.tp}_size_{self.holdout_size}_{self.args.method}.png')
            # plt.show()
            plt.close()

    def plot_confusion_matrix(self, df_conf_mat, itera):
        out_path = f'corr_vis/by_predictions/{self.args.dataset}_tp_{self.tp}_size_{self.holdout_size}_itera_{itera}_{self.args.method}.png'
        plot_heatmap(df_conf_mat, out_path)

    def plot_diagonal_correlation_matrix(self, corr, itera):
        out_path = OUT_DIR + 'corr_vis/by_predictions/'
        mask = np.triu(np.ones_like(corr, dtype=np.bool))
        f, ax = plt.subplots(figsize=(11, 9))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        hmap = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                           square=True, linewidths=.5, cbar_kws={"shrink": .5})
        fig = hmap.get_figure()
        fig.savefig(
            out_path + f'{self.args.dataset}_tp_{self.tp}_itera_{itera}_size_{self.holdout_size}_{self.args.method}.png')
        plt.close(fig)
        # plt.show()


def plot_heatmap(df, out_path, original_map):
    original_map = dict(map(reversed, original_map.items()))
    out_path = OUT_DIR + out_path
    f, ax = plt.subplots(1,1, figsize=(10,12))
    labels = df.index.tolist()
    # map_ = {1: 'sit', 2:'stand', 3:'lie on back', 4:'lie to right', 5:'ascend stairs', 6:'descend stairs',
    #        7: 'stand in elevator', 8: 'move in elevator', 9: 'walk in parking lot', 10:'walk flat at 4 km/h',
    #        11: 'walk inclined at 4 km/h', 12: 'run at 8 km/h', 13: 'stepper exercise', 14: 'cross trainer exercise',
    #        15: 'cycle horizontally', 16: 'cycle vertically', 17: 'row', 18: 'jump', 19: 'play basketball'}
    #
    # true_labels = list(itemgetter(*list(itemgetter(*labels)(original_map)))(
    #                                        map_))
    # replacements = {l1: l2 for l1, l2 in zip(labels, true_labels)}
    # df = df.rename(index = replacements)
    # df.rename(columns = replacements, inplace=True)
    hmap = sns.heatmap(df, ax=ax, annot=False, xticklabels = 1, yticklabels=1)
    hmap.set_yticklabels(hmap.get_yticklabels(),  fontsize=11) # rotation=50,)
    hmap.set_xticklabels(hmap.get_xticklabels(), fontsize=11) # rotation=50, horizontalalignment='right')

    for label in ax.get_xticklabels()[1::2]:
        label.set_visible(False)
    fig = hmap.get_figure()
    plt.xlabel("")
    plt.ylabel("")
    fig.savefig(out_path, bbox_inches='tight')
    plt.tick_params(axis='both', which='major', labelsize=11)
    plt.show()
    plt.close(fig); exit(1)
