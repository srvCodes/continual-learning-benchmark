from matplotlib import pyplot as plt
from train.visualisations import training_visualizer
from matplotlib import pyplot as plt
import matplotlib
from train.visualisations import training_visualizer
import matplotlib.ticker as ticker

plt.tight_layout()
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)

def draw_lines_with_stddev(class_to_means_dict, class_to_stddev_dict, filename, size_val):
    legends = list(class_to_means_dict.keys())
    out_path = training_visualizer.OUT_DIR + 'accuracy_vis/detail_acc/'
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    print(class_to_means_dict)
    batches = [i for i in range(2, len(class_to_means_dict['base']) + 2)]

    for class_id in legends:
        print(class_to_means_dict[class_id])
        ax1.plot(batches, class_to_means_dict[class_id])
    ax1.set_xticks(batches)
    ax1.set_xticklabels(batches)
    ax1.legend(labels=['Base classes', 'Old classes', 'New classes', 'All classes'])
    ax1.set_title("Detailed accuracy comparison")
    fig.savefig(out_path + f'{filename}_size_{size_val}.png')
    plt.show()
    plt.close(fig)


def draw_multiple_lines(method_to_scores_dict, label, size, filename):
    legends = list(method_to_scores_dict.keys())
    out_path = training_visualizer.OUT_DIR + 'accuracy_vis/detail_acc/'
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    batches = [i for i in range(2, len(list(method_to_scores_dict.values())[0]) + 2)]

    for x_label in batches:
        ax1.axvline(x=x_label, linestyle='--', color='grey')
    for method in legends:
        ax1.plot(batches, method_to_scores_dict[method])
    ax1.set_xticks(batches)
    ax1.set_xticklabels(batches)
    ax1.legend(labels=legends)
    ax1.set_title(label)
    fig.savefig(out_path + f'{filename}_holdout_{size}.png')
    # plt.show()
    plt.close(fig)


def draw_multiple_lines_and_plots(base_to_scores_dict, old_to_scores_dict, new_to_scores_dict, all_to_scores_dict, size,
                                  filename, baseline=True):
    legends = list(base_to_scores_dict.keys())
    if baseline:
        labels = ['WA-MDF', 'WA-ADB', 'BiC', 'LUCIR', 'iCaRL', 'ILOS', 'GEM', 'R-EWC', 'MAS', 'LwF']
        colors = ['firebrick', 'green', 'deepskyblue', 'steelblue', 'chocolate', 'gold', 'orangered', 'lightseagreen', 'darkviolet',
                  'indigo', 'darkkhaki', 'dimgray']
    else:
        labels = ['CE', 'LwF', 'R-EWC', 'MAS', 'LUCIR-DIS', 'LUCIR-MR', 'LUCIR-DIS+MR', 'ILOS']
        colors = ['royalblue', 'limegreen', 'gold', 'orangered', 'chocolate', 'teal', 'dimgray', 'mediumorchid' ]
    out_path = training_visualizer.OUT_DIR + 'accuracy_vis/detail_acc/base_old_new/'

    fig, ax1 = plt.subplots(1, 4, figsize=(13,6.5))
    # plt.tick_params(axis='both', which='major', labelsize=13)

    batches = [i for i in range(2, len(list(base_to_scores_dict.values())[0]) + 2)]
    accs = [i for i in range(0, 120, 20)]
    for method, color, label in zip(legends, colors, labels):
        ax1[0].plot(batches, base_to_scores_dict[method], color=color, label=label)
        ax1[1].plot(batches, old_to_scores_dict[method], color=color, label=label)
        ax1[2].plot(batches, new_to_scores_dict[method], color=color, label=label)
        ax1[3].plot(batches, all_to_scores_dict[method], color=color, label=label)
    for ax in ax1:
        ax.set_xticks(batches)
        ax.set_xticklabels(batches)
        ax.set_yticks(accs)
        ax.set_yticklabels(accs)
        if filename == 'twor':
            for label in ax.get_xticklabels()[1::2]:
                label.set_visible(False)
    for x_label in batches:
        for each in ax1:
            each.axvline(x=x_label, linestyle='--', color='aliceblue',
                         zorder=0)  # zorder for shifting line to background
            box = each.get_position()
            each.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax1[0].set_title("Base",fontdict={'fontsize': 16, 'fontweight': 'medium'})
    ax1[1].set_title("Old",fontdict={'fontsize': 16, 'fontweight': 'medium'})
    ax1[2].set_title("New",fontdict={'fontsize': 16, 'fontweight': 'medium'})
    ax1[3].set_title("All",fontdict={'fontsize': 16, 'fontweight': 'medium'})
    lgd = plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', fontsize=13)
    # lgd.get_title().set_fontsize('15')  # legend 'Title' fontsize
    # plt.legend().set_visible(False)

    fig.text(0.5, 0.03, 'No. of tasks', ha='center', va='center', fontsize=13)
    fig.text(0.06, 0.5, 'Micro-F1 (%)', ha='center', va='center', rotation='vertical', fontsize=13)

    """
    Sort the legends based on scores of last task for all classes.
    Source: https://stackoverflow.com/a/27512450/5140684
    """
    # final_task_scores = [each[-1] for each in list(all_to_scores_dict.values())]
    # order = sorted(range(len(final_task_scores)), key=lambda x: final_task_scores[x], reverse=True)
    # handles, labels_ = ax1[3].get_legend_handles_labels()
    # ax1[3].legend([handles[idx] for idx in order], [labels_[idx] for idx in order], bbox_to_anchor=(1, 0.5), loc='center left')
    """End of sorting legends"""
    plt.subplots_adjust(wspace=0.5)
    # plt.text(0.5, 1.08, f"{'DSADS' if filename == 'dsads' else 'WS'}",
    #          horizontalalignment='center',
    #          fontsize=14,
    #          transform=ax1[1].transAxes)
    fig.savefig(out_path + f'{filename}_{"baseline" if baseline else "regularized"}_detailed_acc_holdout_{size}.pdf', bbox_inches='tight', dpi=700)
    plt.show()
    plt.close(fig)

def draw_accs_by_size(size_to_all_accs, filename, key, baseline=False, method='', tp=False):
    legends = list(size_to_all_accs.keys())
    if baseline:
        labels = ['WA-MDF', 'WA-ADB', 'BiC', 'LUCIR', 'iCaRL', 'ILOS', 'GEM']
        colors = ['firebrick', 'green', 'deepskyblue', 'steelblue', 'chocolate', 'gold', 'orangered', 'lightseagreen', 'darkviolet']
    elif tp:
        labels = ['WA-MDF', 'LUCIR', 'ILOS', 'R-EWC']
        colors = ['orangered', 'green',  'gold', 'indigo']
        method_to_label = {'kd_kldiv_wa1': 'WA-MDF', 'cn_lfc_mr': 'LUCIR', 'kd_kldiv_ilos' : 'ILOS', 'ce_online_ewc': 'R-EWC'}
    elif len(method) > 0:
        labels = ['Random', 'Herding', 'Exemplar', 'Boundary', 'FWSR']
        colors = ['firebrick', 'green', 'deepskyblue', 'gold', 'darkviolet',
                  'indigo']
        method_to_label = {'kd_kldiv_wa1': 'WA-MDF', 'cn_lfc_mr': 'LUCIR', 'kd_kldiv_ilos' : 'ILOS', 'ce_online_ewc': 'R-EWC'}
    else:
        labels = ['CE', 'LwF', 'R-EWC', 'MAS', 'LUCIR-DIS', 'LUCIR-MR', 'LUCIR-DIS+MR', 'ILOS']
        colors = ['royalblue', 'limegreen', 'gold', 'orangered', 'chocolate', 'teal', 'dimgray', 'mediumorchid' ]

    out_path = training_visualizer.OUT_DIR + 'accuracy_vis/detail_acc/holdout_sizes/'
    fig, ax1 = plt.subplots()
    # plt.tick_params(axis='both', which='major', labelsize=13)
    batches = [0, 10, 30, 50, 70] if tp else [0]+ [i for i in list(list(size_to_all_accs.values())[0].keys())]
    yticks = [i for i in range(0, 120, 20)]
    order = [100 * (i+1) for i in range(len(list(list(size_to_all_accs.values())[0].values())))]
    for method_, color, label in zip(legends, colors, labels):
        ax1.plot(order, list(size_to_all_accs[method_].values()), color=color, label=label)

    for x_label in order:
        ax1.axvline(x=x_label, linestyle='--', color='aliceblue', zorder=0)  # zorder for shifting line to background
    ax1.set_yticks(yticks)
    plt.locator_params(axis='x', nbins=len(order))
    ax1.set_yticklabels(yticks)
    ax1.set_xticklabels(batches)
    box = ax1.get_position()
    # ax1.set_position([box.x0, box.y0, box.width * 0.9, box.height])

    plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', fontsize=13)
    # ax1.legend().set_visible(False)

    ax1.set_xlabel('Train set size (%)' if tp else 'Holdout size per class', fontsize=13)
    ax1.set_ylabel(f'{"Macro" if key == "macro" else "Micro"}-F1 (%)', fontsize=13)
    if len(method) > 0 or tp:
        plt.title(f"{'DSADS' if filename == 'dsads' else 'WS' if filename == 'ws' else 'HA' if filename == 'hatn6' else 'HAPT' if filename == 'hapt' else 'PAMAP2' if filename == 'pamap' else filename.upper()}"
                  f"{':' + method_to_label[method] if len(method) > 0 else ''}", fontsize=14)

    fig.savefig(out_path + f'{filename}_{key}_{"baseline" if baseline else method if len(method) > 0 else "regularized"}_all_holdouts.pdf', bbox_inches='tight', dpi=700)
    # plt.show()
    plt.close(fig)

def draw_scores_by_task(scores_by_task, filename, methods, replay=True, baseline = True):
    legends = list(methods)
    if baseline:
        labels = ['WA-MDF', 'WA-ADB', 'BiC', 'LUCIR', 'iCaRL', 'ILOS', 'GEM', 'R-EWC', 'MAS', 'LwF']
        colors = ['firebrick', 'green', 'deepskyblue', 'steelblue', 'chocolate', 'gold', 'orangered', 'lightseagreen', 'darkviolet',
                  'indigo', 'darkkhaki', 'dimgray']
    else:
        labels = ['CE', 'LwF', 'R-EWC', 'MAS', 'LUCIR-DIS', 'LUCIR-MR', 'LUCIR-DIS+MR', 'ILOS'] if replay else ['CE', 'LwF', 'R-EWC', 'MAS', 'LUCIR-DIS', 'ILOS']
        colors = ['royalblue', 'limegreen', 'gold', 'orangered', 'chocolate', 'teal', 'dimgray', 'mediumorchid' ] if replay \
            else ['royalblue', 'limegreen', 'gold', 'orangered', 'chocolate',  'mediumorchid' ]
    tasks = [i for i in range(1, len(scores_by_task[0])+1)]
    out_path = training_visualizer.OUT_DIR + 'accuracy_vis/detail_acc/forgetting/'
    fig, ax1 = plt.subplots()
    plt.tick_params(axis='both', which='major', labelsize=13)
    #
    for idx, (scores, color, label) in enumerate(zip(scores_by_task, colors, labels)):
        ax1.plot(tasks, scores, color=color, label=label)

    for x_label in tasks:
        ax1.axvline(x=x_label, linestyle='--', color='aliceblue',
                     zorder=0)  # zorder for shifting line to background
    ax1.set_xticks(tasks)
    ax1.set_xticklabels(tasks)


    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.9, box.height])

    plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', fontsize=13)
    # ax1.legend().set_visible(False)
    ax1.set_xlabel('Incremental task', fontsize=13)
    ax1.set_ylabel(f'Forgetting score', fontsize=13)
    # plt.title(f"{'With rehearsal' if replay else 'Without rehearsal'}", fontsize=14)
    # plt.title(f"{'DSADS' if filename == 'dsads' else 'WS' if filename == 'ws' else 'HA' if filename == 'hatn6' else 'HAPT' if filename == 'hapt' else 'PAMAP2' if filename == 'pamap' else filename.upper()}", fontsize=14)
    fig.savefig(out_path + f'{filename}_forgetting{"_baseline" if baseline else "_regularized"}{"_replay" if replay else "_blank"}.pdf',
                bbox_inches='tight', dpi=700)
    plt.show()
    plt.close(fig)

