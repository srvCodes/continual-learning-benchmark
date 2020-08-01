"""
Author: Saurav Jha
ID: 190029087
"""

import argparse
import time


from train import trainer, result_analyser

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='dsads', type=str,
                    help='Possible values: dsads, pamap, opp, hapt, ws, hatn6, milan,'
                         ' aruba, and twor')
parser.add_argument('--total_classes', default=19, type=int,
                    help='12 for pamap, 12 for hapt, 19 for dsads, 15 for milan, '
                         '11 for aruba, 23 for twor, 9 for ws, 7 for hatn6, 6 for opp')
parser.add_argument('--new_classes', default=2, type=int, help='number of new classes per incremental batch')
parser.add_argument('--base_classes', default=2, type=int, help='number of classes in first batch')
parser.add_argument('--epochs', default=200, type=int, help='number of training epochs: 200 for dsads/pamap')
parser.add_argument('--T', default=2, type=float, help='temperature value for distillation loss')
parser.add_argument('--average_over', default='holdout', type=str,
                    help="whether to average over different holdout sizes: "
                         " 'holdout', different train percents: 'tp'"
                         "or a single run: 'na'")
parser.add_argument('--tp', default=1.0, type=int,
                    help='Fixed train percent to use if "average_over" = "holdout"')
parser.add_argument('--exemp_size', default=6, type=int,
                    help="Fixed holdout size to use if 'average_over' = 'tp'")
parser.add_argument('--method', default='kd_kldiv', type=str,
                    help="distillation method to use: 'ce' for only cross entropy"
                         "'kd_kldiv' for base distillaiton loss with kl divergence "
                         "'kd_kldiv_bic' for Large Scale Incremental Learning, "
                         "'kd_kldiv_wa1' for Maintaining Discrimination and Fairness in Class Incremental Learning,"
                         "'kd_kldiv_wa2' for Adjusting Decision Boundary for Class Imbalanced Learning"
                         " 'cn': cosine norm with basic distillation loss 'cn_lfc': "
                         "cosine normaliztion with less forget constraint as distillation loss, "
                         "'cn_lfc_mr' : cosine norm + less forget constraint + margin ranking loss,"
                         " 'ewc' for elastic weight consolidation with each task having their own importance matrix,"
                         "'online_ewc' for regularised ewc where there will be only one importance matrix across all tasks, "
                         "'lwf': learning without forgetting, 'gem': gradient episodic memory,"
                         " 'agem': averaged gem, 'ce_holdout': cross entropy with memory replay,"
                         "'ce_ewc': EWC with memory replay,"
                         "'ce_lfc': Cross entropy (CE) with less forget constraint, 'ce_mr': CE with margin ranking loss,"
                         "'ce_replaced': CE with ILOS (--replace_new_logits should be set to True for this to work)")
parser.add_argument('--exemplar', default='random', type=str, help="exemplar selection strategy: 'random', 'icarl', "
                                                                   "'kmeans', 'dpp', 'boundary', 'sensitivity' or 'fwsr'")
parser.add_argument('--replace_new_logits', default=False, type=bool, help='if True, replace logits for new class (Incremental Learning in Online Scenario paper)')
parser.add_argument('--wt_init', default=False, type=bool,
                    help="whether to initialize the weights for old classes using "
                         "data stats or not")
parser.add_argument('--weighted', default=False, type=bool,
                    help="whether to weight the new and old class samples or not")
parser.add_argument('--rs_ratio', default=0.7, type=float, help='0 <= rescale ratio <= 1 to use if --weighted is True')
parser.add_argument('--lwf_lamda', default=1.6, type=float, help="loss balance weight for LwF whose higher values favor"
                                                                 " old task performance.")
parser.add_argument('--lamda_base', default=5.0, type=float,
                    help='Base lamda for weighting less forget constraint loss.')
parser.add_argument('--wa2_gamma', default=0.1, type=float, help='Rescaling factor for wa2 method.')
parser.add_argument('--vis', default=False, type=bool, help='visualizing the raw dataset by persons')
parser.add_argument('--tsne_vis', default=False, type=bool,
                    help='tsne visualisations of the intermediate model features')
parser.add_argument('--norm_vis', default=False, type=bool, help='visualising norms of final layer weights by classes '
                                                                 'and by epochs.')
parser.add_argument('--acc_vis', default=False, type=bool, help='visualising accuracies of old and new classes.')
parser.add_argument('-corr_vis', default=False, type=bool, help='correlation heatmaps of classes using raw data '
                                                                'as well as predictions')
parser.add_argument('--exemp_vis', default=False, type=bool, help='help visualising the space occupied by the selected '
                                                                  'exemplars within the class')
parser.add_argument('--reg_coef', default=.2, type=float, help='Regularization coefficient for "online_ewc": a larger '
                                                               'value means less plasticity')
args = parser.parse_args()



def main():
    """
    Main function to train and test. Also for analysing forgetting and accuracy scores
    :return: none
    """
    start_time = time.time()
    model_trainer = trainer.Trainer(args)
    print(f"Total elpased time: {time.time() - start_time}")

    """Uncomment for analysing the saved results in the text files:"""
    # result_analyser.visualize_size_wise_sampling_scores('twor')
    # result_analyser.visualize_tp_wise_sampling_scores('pamap')
    # result_analyser.visualize_size_wise_scores('dsads', baseline=True)
    # result_analyser.visualize_base_old_all_scores('hatn6', baseline=True)
    # result_analyser.visualize_forgetting_measure(filename='hatn6', replay=True, baseline=True)

    """ Uncomment for analysing the forgetting scores, and reports by holdout sizes."""
    filename = 'output_reports/mnist/kd_kldiv_wa1_random_1.0_200'
    # analyser = result_analyser.ResultAnalysis(filename, 30)
    # analyser.parse_text_results()
    # analyser.compute_avg_report_by_sizes()
    # analyser.compute_avg_detailed_accs()
    # analyser.plot_detailed_acc()


if __name__ == "__main__":
    main()
