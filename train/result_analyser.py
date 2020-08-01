from statistics import stdev, mean

import pandas as pd
import pickle
from .visualisations import stability_visualizer
import re

def pickle_reader(filename):
    accuracies_ = pickle.load(open(filename, 'rb'))
    return accuracies_

class ResultAnalysis():
    def __init__(self, filename, seq_len):
        self.pkl_file = filename + '_logs.pkl'
        self.txt_file = filename + '.txt'
        self.seq_len = seq_len
        self.data = pickle_reader(self.pkl_file)
        self.size_to_reports_dict = self.data['reports']
        self.size_to_detailed_acc = self.data['detailed_acc']

    def compute_avg_report_by_sizes(self):
        """
        Function to compute the average over scikit-learn's list of classification_report objects.
        :param size_to_reports_dict: Dictionary with keys = labels indicating tp/holdout sizes,
        values = list of sklearn's classification_report objects
        :return: dict with keys = keys of size_to_reports_dict, values = metrics averaged over the respective sequence of
        values of size_to_reports_dict
        """
        # del self.size_to_reports_dict[4]
        # del self.size_to_reports_dict[6]
        # del self.size_to_reports_dict[8]
        # del self.size_to_reports_dict[10]
        # del self.size_to_reports_dict[15]

        accuracy_by_person = {}
        precision = 3  # round off limit
        for size_key, _list in self.size_to_reports_dict.items():
            print(len(_list), size_key)
            # assert len(_list) == self.seq_len, f"Sequence lengths should be exactly {self.seq_len}."
            # calculate avg over this tp
            by_tp = {key: {'precision': [], 'recall': [], 'f1-score': [], 'support': []} if key != 'accuracy' else []
                     for key, val in _list[0].items()}
            for sequence in _list:
                for key, val in sequence.items():
                    if key != 'accuracy':
                        for metric, score in val.items():
                            by_tp[key][metric].append(score)
                    else:
                        by_tp[key].append(val)

            averaged_dict = {key: {metric: f"{mean(scores) * 100:.{precision}f} +/- {stdev(scores) * 100:.{precision}f}" for
                                   metric, scores in
                                   metric_dict.items()} if key != 'accuracy' else
            f"{mean(metric_dict) * 100:.{precision}f} +/- {stdev(metric_dict) * 100:.{precision}f}" for key, metric_dict in
                             by_tp.items()}
            accuracy_by_person[size_key] = averaged_dict

        for key in self.size_to_reports_dict.keys():
            print(" =================== " * 10)
            print(f"\n\nAveraged classification report for tp/holdout: {key}")
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(pd.DataFrame(accuracy_by_person[key]).transpose())
        return accuracy_by_person

    def compute_avg_report_by_sizes_wo_stddev(self):
        """
        Function to compute the average over scikit-learn's list of classification_report objects.
        :param size_to_reports_dict: Dictionary with keys = labels indicating tp/holdout sizes,
        values = list of sklearn's classification_report objects
        :return: dict with keys = keys of size_to_reports_dict, values = metrics averaged over the respective sequence of
        values of size_to_reports_dict
        """
        # del self.size_to_reports_dict[4]
        # del self.size_to_reports_dict[6]
        # del self.size_to_reports_dict[8]
        # del self.size_to_reports_dict[10]
        # del self.size_to_reports_dict[15]

        accuracy_by_person = {}
        precision = 3  # round off limit
        for size_key, _list in self.size_to_reports_dict.items():
            print(len(_list), size_key)
            # assert len(_list) == self.seq_len, f"Sequence lengths should be exactly {self.seq_len}."
            # calculate avg over this tp
            by_tp = {key: {'precision': [], 'recall': [], 'f1-score': [], 'support': []} if key != 'accuracy' else []
                     for key, val in _list[0].items()}
            for sequence in _list:
                for key, val in sequence.items():
                    if key != 'accuracy':
                        for metric, score in val.items():
                            by_tp[key][metric].append(score)
                    else:
                        by_tp[key].append(val)

            averaged_dict = {key: {metric: mean(scores) * 100 for
                                   metric, scores in
                                   metric_dict.items()} if key != 'accuracy' else
                                    mean(metric_dict) * 100 for key, metric_dict in
                             by_tp.items()}
            accuracy_by_person[size_key] = averaged_dict

        return accuracy_by_person

    def compute_avg_detailed_accs(self):
        self.average_acc = {size_val: {'micro': {}, 'macro':{}} for size_val in self.size_to_detailed_acc.keys()
                            if len(self.size_to_detailed_acc[size_val]['micro']) > 0}
        self.average_std_dev = {size_val: {'micro': {}, 'macro':{}} for size_val in self.size_to_detailed_acc.keys()
                                if len(self.size_to_detailed_acc[size_val]['macro']) > 0}
        for size_val, _dict in self.size_to_detailed_acc.items():
            for key in ['micro', 'macro']:
                for task_num, __list in _dict[key].items():
                    if len(__list) > 0:
                        print(size_val, len(__list))
                        # assert len(__list) == self.seq_len, f"Sequence lengths should be exactly {self.seq_len}."
                        base_, new_, old_, all_ = [], [], [], []
                        for i in range(len(__list)):
                            base_.append(__list[i][0])
                            old_.append(__list[i][1])
                            new_.append(__list[i][2])
                            all_.append(__list[i][3])
                        for key__, val in zip(['base', 'old', 'new', 'all'], [base_, old_, new_, all_]):
                            if key__ not in self.average_acc[size_val][key]:
                                self.average_acc[size_val][key][key__] = [mean(val)]
                                self.average_std_dev[size_val][key][key__] = [stdev(val)]
                            else:
                                self.average_acc[size_val][key][key__].append(mean(val))
                                self.average_std_dev[size_val][key][key__].append(stdev(val))
                    else:
                        continue

    def plot_detailed_acc(self):
        for size_val, _dict in self.average_acc.items():
            for key in ['micro']:
                means, std_devs = [], []
                filename = self.pkl_file.split('/')[1]
                stability_visualizer.draw_lines_with_stddev(_dict[key], self.average_std_dev[size_val][key], filename, size_val)

    def parse_text_results(self):
        """
        Warning: code might be lagging and unoptimized
        :return:
        """
        f = open(self.txt_file, 'r')
        lines = f.readlines()
        f.close()

        task_sequence_lines, task_sequence_lines_idx = [],[]
        task_seq_ID = 'Random sequence of classes:'
        for idx, line in enumerate(lines):
            if line.startswith(task_seq_ID):
                line = line.split(':')[1]
                line = [each.strip() for each in re.sub('[\[\]\n]', '', line).split(',')]
                task_sequence_lines.append(line)
                task_sequence_lines_idx.append(idx)

        sequence_to_previous_task_scores = {i: {} for i in range(0, len(task_sequence_lines))}
        sequence_to_curr_task_scores = {i: {} for i in range(0, len(task_sequence_lines))}

        task_start_line = 0
        for idx_, sequence in enumerate(task_sequence_lines): # go through each task sequence
            task_wise_seq = []
            for idx in range(0, len(sequence), 2):
                temp = []
                for i in range(2):
                    if (idx + i) < len(sequence):
                        temp.append(sequence[idx + i])
                task_wise_seq.append(temp)
            task_cnt = 0
            sequence_limit = task_sequence_lines_idx[idx_]
            task_wise_averaged = []
            task_wise_base_acc = {i: 0 for i in range(0, len(task_wise_seq))}
            cur_task_score = []
            cur_task_to_prev_tasks_dict = {i: {} for i in range(1, len(task_wise_seq))}
            for line_idx in range(task_start_line, sequence_limit):
                cur_line = lines[line_idx]
                if '\t' in cur_line:
                    cur_task = sorted(task_wise_seq[task_cnt])
                    prev_tasks = task_wise_seq[:task_cnt]
                    if any([cur_line.startswith(class_) for class_ in cur_task]) or any([cur_line.startswith(x) for x in [y for each in prev_tasks for y in each]]):
                        # should either start with any of current task or previous task labels
                        f_score = float(cur_line.split('\t')[-2])
                        if any([int(float(cur_line.split('\t')[0].strip())) == int(float(class_)) for class_ in cur_task]):
                            cur_task_score.append(f_score)
                            # print(cur_task_score, task_cnt)
                            if len(cur_task_score) == len(cur_task):
                                task_wise_base_acc[task_cnt] = mean(cur_task_score) # mean of current task scores
                                cur_task_score.clear()
                        # continue
                        elif any([int(float(cur_line.split('\t')[0].strip())) == int(float(x)) for x in [y for each in prev_tasks for y in each]]):
                            class_ = cur_line.split('\t')[0]
                            prev_task_id = [idx for idx, item in enumerate(task_wise_seq[:task_cnt]) if class_ in item][0]
                            if prev_task_id not in cur_task_to_prev_tasks_dict[task_cnt]:
                                cur_task_to_prev_tasks_dict[task_cnt][prev_task_id] = [f_score]
                            else:
                                cur_task_to_prev_tasks_dict[task_cnt][prev_task_id].append(f_score)
                            if len(cur_task_to_prev_tasks_dict[task_cnt][prev_task_id]) == 2:
                                mean_score = mean(cur_task_to_prev_tasks_dict[task_cnt][prev_task_id])
                                cur_task_to_prev_tasks_dict[task_cnt][prev_task_id] = mean_score
                                if mean_score > task_wise_base_acc[prev_task_id]:
                                    # this is the maximum knowledge gained about the task
                                    task_wise_base_acc[prev_task_id] = mean_score
                elif cur_line.startswith(' -----'):
                    task_cnt += 1
                    task_start_line = sequence_limit + 3
            sequence_to_previous_task_scores[idx_].update(cur_task_to_prev_tasks_dict)
            sequence_to_curr_task_scores[idx_].update(task_wise_base_acc)

        forgetting_scores = calculate_forgetting(sequence_to_previous_task_scores, sequence_to_curr_task_scores)
        return forgetting_scores

def calculate_forgetting(sequence_to_previous_task_scores, sequence_to_max_task_scores):
    forgetting_scores = {seq_id: {} for seq_id in sequence_to_previous_task_scores.keys()}
    for seq_id, prev_task_to_scores_dict in sequence_to_previous_task_scores.items():
        for task_id, prev_task_scores in prev_task_to_scores_dict.items():
            # F_k = [sequence_to_max_task_scores[seq_id][prev_task_id] - score for prev_task_id, score in
            #        prev_task_scores.items() if sequence_to_max_task_scores[seq_id][prev_task_id] > 0]
            F_k = [1.0 - score / sequence_to_max_task_scores[seq_id][prev_task_id]  for prev_task_id, score in
                   prev_task_scores.items() if sequence_to_max_task_scores[seq_id][prev_task_id] > 0] # consider only those tasks that the model remembers
            forgetting_scores[seq_id][task_id] = mean(F_k)
    df = pd.DataFrame.from_dict(forgetting_scores).T
    df.columns = ['Task ' + str(i+1) for i in range(len(forgetting_scores[0]))]
    return df

def visualize_base_old_all_scores(filename, baseline=True):
    size_to_base_accs = {}
    size_to_old_accs = {}
    size_to_new_accs = {}
    size_to_all_accs = {}
    methods = ['kd_kldiv_wa1', 'kd_kldiv_wa2', 'kd_kldiv_bic', 'cn_lfc_mr', 'kd_kldiv_icarl',  'kd_kldiv_ilos', 'gem',
                'online_ewc', 'mas', 'lwf'] if baseline else \
        ['ce_holdout', 'kd_kldiv', 'ce_ewc', 'ce_mas', 'ce_lfc', 'ce_mr', 'cn_lfc_mr', 'ce_replaced_ilos']
    analyzer_objects = []
    for method in methods:
        if filename == 'hapt' and method == 'online_ewc':
            method = 'ewc' # ewc performs better for hapt
        path_name = f"output_reports/{filename}/{method}{'_' if 'icarl' in method else '_random_'}1.0_15"
        analyser = ResultAnalysis(path_name, 30)
        analyser.compute_avg_detailed_accs()
        for size_ in [2, 4, 6]:
            if method in ['mas', 'ewc', 'online_ewc', 'lwf', 'ce']:
                # these don't use memory replay so copy the scores from holdout size 15 to all other sizes
                val = analyser.average_acc[15]
            else:
                val = analyser.average_acc[size_]
            if size_ in size_to_base_accs:
                size_to_base_accs[size_][method] = val['micro']['base']
                size_to_old_accs[size_][method] = val['micro']['old']
                size_to_new_accs[size_][method] = val['micro']['new']
                size_to_all_accs[size_][method] = val['micro']['all']
            else:
                print(size_, val, method)
                size_to_base_accs[size_] = {method: val['micro']['base']}
                size_to_old_accs[size_] = {method: val['micro']['old']}
                size_to_new_accs[size_] = {method: val['micro']['new']}
                size_to_all_accs[size_] = {method: val['micro']['all']}

    for size_ in [6]:
        stability_visualizer.draw_multiple_lines_and_plots(size_to_base_accs[size_], size_to_old_accs[size_],
                                                           size_to_new_accs[size_],
                                                           size_to_all_accs[size_], size_, filename, baseline=baseline)

def visualize_size_wise_scores(filename, baseline=True):
    size_to_all_accs = {}
    key = 'micro'
    methods_replay = ['ce_holdout', 'kd_kldiv', 'ce_ewc', 'ce_mas', 'ce_lfc', 'ce_mr', 'cn_lfc_mr', 'ce_replaced_ilos']
    methods_baseline = ['kd_kldiv_wa1', 'kd_kldiv_wa2', 'kd_kldiv_bic', 'cn_lfc_mr', 'kd_kldiv_icarl', 'kd_kldiv_ilos',
                        'gem']
    methods = methods_baseline if baseline else methods_replay

    analyzer_objects = []
    for method in methods:
        print(method)
        path_name = f"output_reports/{filename}/{method}{'_' if 'icarl' in method else '_random_'}1.0_15"

        analyser = ResultAnalysis(path_name, 30)
        acc = analyser.compute_avg_report_by_sizes_wo_stddev()
        for size_ in [2,4,6,8,10,15]:
            if key == 'micro':
                if method not in size_to_all_accs:
                    size_to_all_accs[method] = {size_: acc[size_]['accuracy']}
                else:
                    size_to_all_accs[method][size_] = acc[size_]['accuracy']
            elif key == 'macro':
                if method not in size_to_all_accs:
                    size_to_all_accs[method] = {size_: acc[size_]['macro avg']['f1-score']}
                else:
                    size_to_all_accs[method][size_] = acc[size_]['macro avg']['f1-score']

    stability_visualizer.draw_accs_by_size(size_to_all_accs, filename, key=key, baseline=baseline)

def visualize_size_wise_sampling_scores(filename, tp=False):
    key = 'micro'
    methods = ['kd_kldiv_wa1', 'cn_lfc_mr', 'kd_kldiv_ilos', 'ce_online_ewc']
    lr = {'pamap': '0.01', 'dsads': '0.01', 'twor': '0.1', 'milan': '0.01'}
    samplings = ['random', 'icarl', 'kmeans', 'boundary', 'fwsr']
    sizes = [2, 4, 6, 8, 10, 15] if not tp else [0.4, 0.6, 0.8, 1.0]
    for method in methods:
        size_to_all_accs = {}
        path_base = f"output_reports/{filename}/{method}"
        for sampling in samplings:
            lr_ = str(0.001) if filename == 'twor' and method in ['ce_online_ewc'] else str(lr[filename])
            path_name = path_base
            path_name += f"_tp_{lr_}_random_{1.0}_6" if tp else f"_{sampling}_1.0_15"

            analyser = ResultAnalysis(path_name, 30)
            acc = analyser.compute_avg_report_by_sizes_wo_stddev()
            for size_ in sizes:
                if key == 'micro':
                    if sampling not in size_to_all_accs:
                        size_to_all_accs[sampling] = {size_: acc[size_]['accuracy']}
                    else:
                        size_to_all_accs[sampling][size_] = acc[size_]['accuracy']
                elif key == 'macro':
                    if sampling not in size_to_all_accs:
                        size_to_all_accs[sampling] = {size_: acc[size_]['macro avg']['f1-score']}
                    else:
                        size_to_all_accs[sampling][size_] = acc[size_]['macro avg']['f1-score']
        stability_visualizer.draw_accs_by_size(size_to_all_accs, filename, key=key, method=method)


def visualize_tp_wise_sampling_scores(filename):
    key = 'macro'
    methods = ['kd_kldiv_wa1', 'cn_lfc_mr', 'kd_kldiv_ilos', 'ce_online_ewc']
    lr = {'pamap': '0.01', 'dsads': '0.01', 'twor': '0.1', 'milan': '0.01'}
    sizes = [0.1429, 0.4286, 0.7143, 1.0]
    size_to_all_accs = {}

    for method in methods:
        path_base = f"output_reports/{filename}/{method}"
        path_name = path_base
        path_name += f"__tprandom_{1.0}_6"

        analyser = ResultAnalysis(path_name, 30)
        acc = analyser.compute_avg_report_by_sizes_wo_stddev()
        for size_ in sizes:
            if key == 'micro':
                if method not in size_to_all_accs:
                    size_to_all_accs[method] = {size_: acc[size_]['accuracy']}
                else:
                    size_to_all_accs[method][size_] = acc[size_]['accuracy']
            elif key == 'macro':
                if method not in size_to_all_accs:
                    size_to_all_accs[method] = {size_: acc[size_]['macro avg']['f1-score']}
                else:
                    size_to_all_accs[method][size_] = acc[size_]['macro avg']['f1-score']
    stability_visualizer.draw_accs_by_size(size_to_all_accs, filename, key=key, tp=True)

def visualize_forgetting_measure(filename, replay=True, baseline=True):
    """

    :param filename:
    :param replay: uses memory replay or not; only in case of baseline=False
    :param baseline:
    :return:
    """
    methods_replay = ['ce_holdout', 'kd_kldiv', 'ce_ewc', 'ce_mas', 'ce_lfc', 'ce_mr', 'cn_lfc_mr', 'ce_replaced_ilos']
    methods_blank = ['ce', 'lwf', 'online_ewc', 'mas', 'blank_ce_lfc', 'blank_ce_replaced_ilos']
    methods_baseline = ['kd_kldiv_wa1', 'kd_kldiv_wa2', 'kd_kldiv_bic', 'cn_lfc_mr', 'kd_kldiv_icarl',  'kd_kldiv_ilos', 'gem',
                'online_ewc', 'mas', 'lwf']
    if baseline:
        methods = methods_baseline
    else:
        methods = methods_replay if replay else methods_blank
    all_forgetting = []
    for idx, method in enumerate(methods):
        if filename == 'hapt' and method == 'online_ewc':
            method = 'ewc' # ewc performs better for hapt
        path_name = f"output_reports/{filename}/{method}{'_' if 'icarl' in method else '_random_'}1.0_"
        if baseline:
            if method in ['mas', 'ewc', 'online_ewc', 'lwf', 'ce']:
                # these don't use memory replay so copy the scores from holdout size 15 to all other sizes
                path_name = path_name + '15'
            else:
                path_name = path_name + '6'
        else:
            if replay:
                path_name = path_name + '6'
            else:
                path_name = path_name + '15' if method not in ['ce', 'lwf'] else path_name + '6'
        analyser = ResultAnalysis(path_name, 30)
        forgetting_scores = analyser.parse_text_results()
        all_forgetting.append(forgetting_scores.mean().values)

    stability_visualizer.draw_scores_by_task(all_forgetting, filename, methods, replay=replay, baseline=baseline)