import math
import pickle
import random
from collections import Counter
from copy import deepcopy
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from models import basic_model, modified_linear
from train import exemplar_selection, architecture_update, prediction_analyzer, customized_distill_trainer
from train.compute_cosine_features import compute_features
from train.gem_solver import project2cone2
from train.visualisations import training_visualizer
from utils import data_handler

OUT_PATH = 'output_reports/'
PATIENCE = 15
TRAIN_PERCENTAGES = [0.1429 , 0.4286, 0.7143, 1.0]
HOLDOUT_SIZES = [2, 4, 6, 8, 10, 15] # [200] for permuted_mnist
# TRAIN_PERCENTAGES.reverse()
# HOLDOUT_SIZES.reverse()
SEEDS = [i for i in range(0, 30)]
BATCH_SIZE = 20
LR = 0.01 # 0.0001 for permuted_mnist
OPTIMIZER_STEP_AFTER = 50 # 0 for permuted_mnist
STEP_SIZE = 50 # 1 for permuted_mnist
WEIGHT_DECAY_RATE = 1e-4 # 0 for permuted mnist

def write_to_report_and_print(outfile, string_result):
    file = open(outfile, 'a')
    file.write('\n' + string_result)
    file.close()
    print(string_result)


class Trainer():
    """
    Class for handling the training and testing processes.
    """
    def __init__(self, args):
        """
        Class initializer.
        :param args: the command line arguments
        """
        self.args = args
        if self.args.average_over == 'na':
            # compute everything for a single run
            tp, seed_val = self.args.tp, 10
            self.list_of_classification_reports = {tp: []}
            self.error_stats = {tp: {'e_n': [], 'e_o': [], 'e_o_n': [], 'e_o_o': []}}
            handler = self.get_dataset_handler(self.args.dataset, self.args.base_classes,
                                               self.args.new_classes,
                                               tp, seed_val, self.args.vis,
                                               keep_val=any([x in self.args.method for x in ['bic']]))
            self.initialize_and_train(seed_val, handler)
        else:
            if self.args.average_over == 'tp':
                """
                Compute everything for different train percentages of data: 10, 30, 50 and 70% (test size = 30% for all)
                Holdout size remains fixed
                """
                self.list_of_classification_reports = {i: [] for i in TRAIN_PERCENTAGES}
                self.error_stats = {i: {'e_n': [], 'e_o': [], 'e_o_n': [], 'e_o_o': []} for i in TRAIN_PERCENTAGES}
                self.detailed_accuracies = {i: {'micro': {j: [] for j in range(1, self.args.total_classes)},
                                                'macro': {j: [] for j in range(1, self.args.total_classes)}} for i in
                                            TRAIN_PERCENTAGES}
            elif self.args.average_over == 'holdout':
                # compute everything for different holdout sizes for old tasks; train percent remains fixed
                self.list_of_classification_reports = {i: [] for i in HOLDOUT_SIZES}
                self.error_stats = {i: {'e_n': [], 'e_o': [], 'e_o_n': [], 'e_o_o': []} for i in HOLDOUT_SIZES}
                self.detailed_accuracies = {i: {'micro': {j: [] for j in range(1, self.args.total_classes)},
                                                'macro': {j: [] for j in range(1, self.args.total_classes)}} for i in
                                            HOLDOUT_SIZES}
            for handler, size_val, seed_val in self.instantiate_trainers():
                self.initialize_and_train(seed_val, handler, size_val=size_val)
                if seed_val == SEEDS[-1]:
                    assert len(self.base_task_times) == len(self.all_task_times) == len(self.old_task_times), \
                        "Time lengths do not match"
                    cost_analyzer = f"Base task time: {sum(self.base_task_times) / len(self.base_task_times)}s\n" \
                                    f"Old task time (averaged) = {sum(self.old_task_times) / len(self.old_task_times)}s\n" \
                                    f"Overall task time (averaged) = {sum(self.all_task_times) / len(self.all_task_times)}s\n" \
                                    f"Exemplar selection time = {sum(self.exemp_task_times) / len(self.exemp_task_times)}"
                    self.report_costs(cost_analyzer)

    def instantiate_trainers(self):
        """
        Function to instantiate the data handler class by iterating over train percentages / holdout sizes and random seeds.
        :return: the data handler instance, train percent / holdout size, and the seed value
        """
        self.base_task_times, self.old_task_times, self.all_task_times, self.exemp_task_times = [], [], [], []
        if self.args.average_over == 'tp':
            for tp in TRAIN_PERCENTAGES:
                for seed in SEEDS:
                    handler = self.get_dataset_handler(self.args.dataset, self.args.base_classes,
                                                       self.args.new_classes,
                                                       tp, seed, self.args.vis, self.args.corr_vis,
                                                       keep_val=any([x in self.args.method for x in ['bic']]))
                    yield handler, tp, seed
        elif self.args.average_over == 'holdout':
            for holout_size in HOLDOUT_SIZES:
                for seed in SEEDS:
                    handler = self.get_dataset_handler(self.args.dataset, self.args.base_classes, self.args.new_classes,
                                                       self.args.tp, seed, self.args.vis, self.args.corr_vis,
                                                       keep_val=any([x in self.args.method for x in ['bic']]))
                    yield handler, holout_size, seed

    def initialize_and_train(self, seed, handler, size_val=None):
        """
        Function for training and logging.
        :param seed: random seed value for reproducibility
        :param handler: the dataset handler instance
        :param size_val: the holdout / train percentage size
        :return: None
        """
        self.initializer(seed, handler, size_val)
        self.train(BATCH_SIZE, LR)
        self.dump_scores()

    def initializer(self, seed_val, handler, size_val=None):
        """
        Defines the dataset and method-specific objects for the class.
        :param seed_val: the random seed value for reproducibility
        :param handler: the dataset handler instance
        :param size_val: train percent / holdout size
        :return: NA
        """
        self.seed_randomness(seed_val)
        self.train_percent = size_val if self.args.average_over == 'tp' else self.args.tp
        self.holdout_size = size_val if self.args.average_over == 'holdout' else self.args.exemp_size
        self.dataset = handler
        self.seen_cls = 0
        self.input_dim = self.dataset.getInputDim()
        self.max_size = self.holdout_size * self.args.total_classes
        self.init_out_files()
        self.original_mapping = self.dataset.get_reversed_original_label_maps()
        self.label_map = dict(map(reversed, self.dataset.label_map.items()))
        self.visualizer = training_visualizer.Visualizer(self.label_map, self.original_mapping, self.args,
                                                         self.dataset.num_tasks, self.train_percent, self.holdout_size)
        # self.visualizer.plot_tsne(np.array(self.dataset.train_data), self.dataset.train_labels, test=True); exit(1)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.bias_layers = [basic_model.BiasLayer(self.device).to(self.device) for i in range(self.dataset.num_tasks)] if \
            'bic' in self.args.method else None
        if any([x in self.args.method for x in ['gem', 'ewc', 'mas']]) or 'permuted' in self.args.dataset:
            self.model = basic_model.Net(self.input_dim, self.args.total_classes, self.args.dataset).to(self.device)
            self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
            if any([x in self.args.method for x in ['ewc', 'mas']]):
                self.n_fisher_sample = None
                if any([x in self.args.method for x in ['online_ewc', 'mas']]):
                    self.regularization_terms = {}
            if 'gem' in self.args.method:
                # define memories for storing task to gradient vectors and cache for all tasks
                self.task_grads = {}
                self.task_mem_cache, self.task_memory = {}, {}

    def init_out_files(self):
        """
        Define the output file paths for logging
        :return: NA
        """
        self.outp = OUT_PATH + self.args.dataset + f"/{self.args.method}{'_ilos_' if self.args.replace_new_logits else '_wt_init_' if self.args.wt_init else '_'}" \
                                                   f"{'_' + self.args.average_over if self.args.average_over == 'tp' else ''}" \
                                                   f"{self.args.exemplar}_{str(self.train_percent)}_{self.holdout_size}"
        self.outfile = self.outp + '.txt'
        self.stats_file = OUT_PATH + self.args.dataset + f"/{self.args.method}" \
                                                         f"{'_ilos_' if self.args.replace_new_logits else '_wt_init_' if self.args.wt_init else '_'}" \
                                                         f"{self.args.exemplar}_stats.txt"

    def dump_scores(self):
        """
        Method to dump the accuracy scores, error values, and detailed accuracies for different sizes.
        :return: NA
        """
        to_dump = {'reports': self.list_of_classification_reports, 'errors': self.error_stats,
                   'detailed_acc': self.detailed_accuracies}
        pickle.dump(to_dump, open(self.outp + '_logs.pkl', 'wb'))

    @staticmethod
    def seed_randomness(seed_value):
        np.random.seed(seed_value)

    @staticmethod
    def get_dataset_handler(dataname, nb_cl, new_cl, tp, seed, vis, corr_vis, keep_val):
        handler = data_handler.DataHandler(dataname, nb_cl, new_cl,
                                           tp, seed, vis, corr_vis, keep_val)
        return handler

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def set_gradients_of_bias_layers(self, bool_val):
        for each in self.bias_layers:
            each.set_grad(bool_val)

    def set_gradients_of_model_layers(self, bool_val):
        for name, param in self.model.named_parameters():
            param.requires_grad = bool_val

    def update_model(self, itera, num_new_classes):
        """
        Method for extending the previous model with new larger number of classes
        :param itera: incremental batch no.
        :param num_new_classes: no. of new classes
        :return: lambda_mult value (to be used for LUCIR)
        """
        if itera > 0:
            self.previous_model = deepcopy(self.model)
            self.previous_model.eval()
        lamda_mult = None
        if 'cn' in self.args.method:
            if itera == 0:
                self.model = basic_model.Net(self.input_dim, num_new_classes, self.args.dataset,
                                             cosine_liner=True)
            else:
                if itera == 1:
                    in_features = self.model.fc.in_features
                    out_features = self.model.fc.out_features
                    new_fc = modified_linear.SplitCosineLinear(in_features, out_features,
                                                               num_new_classes)
                    new_fc.fc1.weight.data = self.model.fc.weight.data[:self.seen_cls - self.args.new_classes]
                    if not self.args.wt_init:
                        # do normal initialisation
                        old_weights = self.model.fc.weight.data[:self.seen_cls - num_new_classes].cpu().detach().numpy()
                        w_mean = np.mean(old_weights, axis=0)
                        w_std = np.std(old_weights, axis=0)
                        new_weights = np.pad(old_weights, ((0, self.args.new_classes), (0, 0)), mode="constant", constant_values=0)
                        for i in reversed(range(self.args.new_classes)):
                            for j in range(new_weights.shape[1]):
                                new_weights[new_weights.shape[0] - 1 - i][j] = np.random.normal(w_mean[j], w_std[j])
                        new_fc.fc2.weight.data = torch.nn.Parameter(torch.from_numpy(new_weights[-self.args.new_classes:]))
                    new_fc.sigma.data = self.model.fc.sigma.data[:self.seen_cls - num_new_classes]
                    self.model.fc = new_fc
                else:
                    in_features = self.model.fc.in_features
                    out_features1 = self.model.fc.fc1.out_features
                    out_features2 = self.model.fc.fc2.out_features
                    new_fc = modified_linear.SplitCosineLinear(in_features, out_features1 + out_features2,
                                                               num_new_classes)
                    new_fc.fc1.weight.data[:out_features1] = self.model.fc.fc1.weight.data
                    new_fc.fc1.weight.data[out_features1:] = self.model.fc.fc2.weight.data
                    if not self.args.wt_init:
                        # do normal initialisation
                        old_weights = new_fc.fc1.weight.data.cpu().detach().numpy()
                        w_mean = np.mean(old_weights, axis=0)
                        w_std = np.std(old_weights, axis=0)
                        new_weights = np.pad(old_weights, ((0, self.args.new_classes), (0, 0)), mode="constant",
                                             constant_values=0)
                        for i in reversed(range(self.args.new_classes)):
                            for j in range(new_weights.shape[1]):
                                new_weights[new_weights.shape[0] - 1 - i][j] = np.random.normal(w_mean[j], w_std[j])
                        new_fc.fc2.weight.data = torch.nn.Parameter(
                            torch.from_numpy(new_weights[-self.args.new_classes:]))
                    new_fc.sigma.data = self.model.fc.sigma.data
                    self.model.fc = new_fc
        else:
            if itera == 0:
                self.model = basic_model.Net(self.input_dim, num_new_classes, self.args.dataset,
                                             lwf=self.args.method == 'lwf')
                if self.args.method == 'lwf':
                    self.model.apply(architecture_update.kaiming_normal_init)
            elif 'permuted' not in self.args.dataset:
                self.model = architecture_update.update_model(basic_model.Net, self.model, self.input_dim,
                                                              self.args.dataset,
                                                              num_new_classes,
                                                              device=self.device, lwf=self.args.method == 'lwf')
                assert architecture_update.check_model_integrity(self.previous_model, self.model) == True, \
                    print("Model integrity violated on update!")
        self.model.to(self.device)
        if itera > 0:
            old_classes_num = self.seen_cls - num_new_classes
            lamda_mult = old_classes_num * 1.0 / num_new_classes
            lamda_mult = self.args.lamda_base * math.sqrt(lamda_mult)
        return lamda_mult

    def imprint_new_weights(self, num_new_classes, train_xs, train_ys):
        """
        Method for initializing the weights of the new neurons for FC layer based on stats of previous model
        :param num_new_classes
        :param train_xs: all train data
        :param train_ys: all train labels
        :return: NA
        """
        print(f"initializing weights for new classes:")
        old_embedding_norm = self.model.fc.fc1.weight.data.norm(dim=1, keepdim=True) if 'cn' in self.args.method else \
            self.model.fc.weight.data[:self.seen_cls - num_new_classes].norm(dim=1, keepdim=True)
        avg_old_embedding_norm = torch.mean(old_embedding_norm, dim=0).cpu().type(torch.DoubleTensor)
        num_features = self.model.fc.in_features
        novel_embedding = torch.zeros((num_new_classes, num_features))
        feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
        for cls_idx in range(self.seen_cls - num_new_classes, self.seen_cls):
            filtered_val_x = [x for x, y in zip(train_xs, train_ys) if y == cls_idx]
            filtered_val_y = np.zeros(len(filtered_val_x))
            evalloader = torch.utils.data.DataLoader(basic_model.Dataset(filtered_val_x, filtered_val_y), batch_size=5,
                                                     shuffle=False)
            cls_features = compute_features(feature_extractor, cls_idx, evalloader, len(evalloader.dataset),
                                            num_features)
            norm_features = F.normalize(torch.from_numpy(cls_features), p=2, dim=1)
            cls_embedding = torch.mean(norm_features, dim=0)
            prod = F.normalize(cls_embedding, p=2, dim=0) * avg_old_embedding_norm
            novel_embedding[cls_idx - self.seen_cls + num_new_classes] = prod
        if 'cn' in self.args.method:
            self.model.fc.fc2.weight.data = novel_embedding.to(self.device)
        else:
            self.model.fc.weight.data[self.seen_cls - num_new_classes:] = novel_embedding.to(self.device)

    def count_parameters(self):
        """Source: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9?u=saurav_jha"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        cnt_bias_corr_params = 0
        if 'bic' in self.args.method:
            for each in self.bias_layers:
                cnt_bias_corr_params += sum(p.numel() for p in each.parameters())
        return total_params, trainable_params, cnt_bias_corr_params

    def report_costs(self, cost_analyzer):
        """
        Write the cost (memory and time) logs to the output file
        :param cost_analyzer: string to write
        :return: NA
        """
        total_params, trainable_params, bias_corr_params = self.count_parameters()
        cost_analyzer += f"Total params: {total_params}, trainable_params: {trainable_params}"
        if 'bic' in self.args.method:
            cost_analyzer += f" Bias correction layer params: {bias_corr_params}"
        cost_analyzer += "\n"
        write_to_report_and_print(self.stats_file, cost_analyzer)

    def build_cache_memory(self, itera, train_loader, num_samples):
        """
        Exemplar selector for GEM and AGEM
        :param itera: incremental batch no.
        :param train_loader: data loader for train set
        :param num_samples: no. of samples to keep for old tasks
        :return: NA
        """
        for task_id in self.task_memory.keys():
            self.task_memory[task_id] = self.task_memory[task_id][:num_samples]

        randind = torch.randperm(len(train_loader.dataset))[:num_samples]
        self.task_memory[itera] = []
        for ind in randind:
            self.task_memory[itera].append(train_loader.dataset[ind])

        for t, mem in self.task_memory.items():
            mem_loader = torch.utils.data.DataLoader(mem, batch_size=len(mem), shuffle=False)
            assert len(mem_loader) == 1, 'The length of mem_loader should be 1'
            for i, (mem_input, mem_target) in enumerate(mem_loader):
                mem_input = mem_input.float().to(self.device)
                mem_target = mem_target.view(-1).to(self.device)
            self.task_mem_cache[t] = {'data': mem_input, 'target': mem_target, 'task': t}

    def train(self, batch_size, lr, norm_visualizer=False):
        """
        Function to train the model over incremental batches.
        :return: detailed accuracies over all batches.
        """
        criterion = torch.nn.CrossEntropyLoss()
        all_test_data, all_test_labels = [], []
        test_accs = []
        task_times, exemp_times = [], []
        if self.args.method not in ['agem']:
            exemp = exemplar_selection.Exemplar(self.args.dataset, self.max_size, self.args.total_classes,
                                                self.input_dim,
                                                self.label_map,
                                                self.original_mapping, self.outfile, self.device)

        for itera in range(self.dataset.num_tasks):
            to_write = ""
            train, val, test = self.dataset.getNextClasses(itera)
            train_x, train_y = zip(*train)
            test_x, test_y = zip(*test)
            all_test_data.extend(test_x)
            all_test_labels.extend(test_y)
            num_new_classes = len(self.dataset.classes_by_groups[itera])  # because the last group classes might not contain self.new_classes
            print(f"nm_new_cls: {num_new_classes}")
            start_time_exemp = timer()
            if 'gem' not in self.args.method:
                train_xs, train_ys = exemp.get_exemplar_train()  # get old task classes from holdout
                val_data = None
                if any([x in self.args.method for x in ['bic']]):
                    val_x, val_y = zip(*val)
                    exemp.update(self.args.exemplar, num_new_classes, itera, self.args.exemp_vis, (train_x, train_y),
                                 (val_x, val_y))
                    # validation data for building fisher matrix in ewc should contain only previous classes
                    val_xs, val_ys = exemp.get_exemplar_val()
                    if itera > 0:
                        val_data = torch.utils.data.DataLoader(basic_model.Dataset(val_xs, val_ys),
                                                               batch_size=batch_size,
                                                               shuffle=True)
                else:
                    exemp.update(self.args.exemplar, num_new_classes, itera, self.args.exemp_vis, (train_x, train_y))
                self.seen_cls = exemp.get_cur_cls()  # total num of classes seen so far

                if self.args.weighted and self.args.rs_ratio > 0 and itera > 0:
                    scale_factor = (len(train_x) * self.args.rs_ratio) / (len(train_xs) * (1 - self.args.rs_ratio))
                    rs_sample_weights = np.concatenate((np.ones(len(train_xs)) * scale_factor, np.ones(len(train_x))))
                    rs_num_samples = int(len(train_x) / (1 - self.args.rs_ratio))
                    print(f"X_train: {len(train_x)}, X_protoset: {len(train_xs)}, rs_num_samples: {rs_num_samples}")
                train_xs.extend(train_x)
                train_ys.extend(train_y)
            else:
                self.seen_cls += num_new_classes
                print([self.original_mapping[i] for i in self.dataset.classes_by_groups[itera]])
                to_write += f"\nNew classes: {[self.original_mapping[i] for i in self.dataset.classes_by_groups[itera]]}"
                if itera > 0:
                    to_write += f", Old classes: {[self.original_mapping[each] for i in self.dataset.classes_by_groups[:itera] for each in i]}"

            to_write += f'\nNo. of new classes: {num_new_classes}, seen classes: {self.seen_cls}, ' \
                        f'total classes: {self.args.total_classes}\n'

            if self.args.method in ['online_ewc', 'ewc', 'gem', 'agem', 'mas', 'ce', 'lwf', 'agem', 'gem',
                                    'blank_ce_mr', 'blank_ce_lfc', 'blank_ce_replaced', 'blank_ce_lfc_mr']:
                    # or self.args.replace_new_logits:
                # training data should only comprise of new task instances
                train_data = torch.utils.data.DataLoader(basic_model.Dataset(train_x, train_y), batch_size=batch_size,
                                                         shuffle=True)
                # if self.args.replace_new_logits and itera > 0:
                #     # balanced data for two-step learning
                #     train_xs_balanced, train_ys_balanced = exemp.get_exemplar_train()
                #     train_data_balanced = torch.utils.data.DataLoader(basic_model.Dataset(train_xs_balanced, train_ys_balanced),
                #                                                       batch_size=batch_size, shuffle=True)
            else:
                # train data will be a blend of new and old tasks with new tasks as majority
                if itera > 0 and self.args.rs_ratio > 0 and self.args.weighted and scale_factor > 1:
                    index1 = np.where(rs_sample_weights > 1)[0]
                    index2 = np.where(np.array(train_ys) < self.seen_cls - num_new_classes)[0]
                    assert ((index1 == index2).all())
                    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(rs_sample_weights, rs_num_samples)
                    train_data = torch.utils.data.DataLoader(basic_model.Dataset(train_xs, train_ys),
                                                             batch_size=batch_size,
                                                             shuffle=False, sampler=train_sampler, drop_last=False)
                else:
                    train_data = torch.utils.data.DataLoader(basic_model.Dataset(train_xs, train_ys),
                                                             batch_size=batch_size,
                                                             shuffle=True, drop_last=False)
            end_time_exemp = timer() - start_time_exemp
            test_data_all = torch.utils.data.DataLoader(basic_model.Dataset(all_test_data, all_test_labels),
                                                        batch_size=batch_size,
                                                        shuffle=False)
            to_write += f'\nInremental batch: {itera}, Size of train set: {len(train_data.dataset)}, validation: {len(val)}, ' \
                        f'test set: {len(test_data_all.dataset)}\n'
            write_to_report_and_print(self.outfile, to_write)

            test_acc = []
            if not any([x in self.args.method for x in ['gem', 'ewc']]) or 'permuted' not in self.args.dataset:
                # since the model is fixed for agem, no need to update final layer dimensions
                cur_lamda = self.update_model(itera, num_new_classes)
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=WEIGHT_DECAY_RATE)
            scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=0.7)

            if itera > 0 and self.args.method not in ['ce', 'ce_holdout', 'ce_ewc', 'online_ewc', 'ewc', 'gem', 'agem', 'mas', 'ce_mas', 'ce_online_ewc']:
                if self.args.wt_init:
                    self.imprint_new_weights(num_new_classes, train_x, train_y)
                # initialize trainer object for distillation loss (not required for ce and ewc as they can be trained normally)
                custom_trainer = customized_distill_trainer.CustomizedTrainer(self.args, itera, self.seen_cls,
                                                                              train_data, self.model,
                                                                              self.previous_model,
                                                                              cur_lamda, self.bias_layers,
                                                                              self.dataset.label_map,
                                                                              self.dataset.classes_by_groups,
                                                                              self.device,
                                                                              self.visualizer)

            last_improvement_epoch, min_loss = 0, 99999
            if norm_visualizer:
                epoch_to_norm_dict = {i: {} for i in range(self.args.epochs)}

            start_time_train = timer()
            for epoch in range(self.args.epochs):
                self.model.train()
                if epoch > OPTIMIZER_STEP_AFTER:
                    scheduler.step()
                cur_lr = self.get_lr(optimizer)
                print(f"---" * 50 + f"\nEpoch: {epoch} Current Learning Rate : {cur_lr}")

                if itera == 0:
                    train_loss = self.normal_training(train_data, criterion, optimizer, itera)
                elif itera > 0:
                    if 'gem' in self.args.method:
                        train_loss = self.gem_training(train_data, criterion, optimizer, itera)
                    elif self.args.method in ['ce', 'ce_holdout', 'ewc', 'online_ewc', 'ce_ewc', 'mas', 'ce_mas', 'ce_online_ewc'] \
                            and not self.args.replace_new_logits:
                        # perform normal training with or without ewc penalty using only new task data at each step
                        train_loss = self.normal_training(train_data, criterion, optimizer, itera)
                    else:
                        print("size: ========= ", len(train_data.dataset))
                        new_class_averaged = self.compute_average_vecs(train_x, train_y) if self.args.weighted else None
                        train_loss = custom_trainer.distill_training(optimizer, num_new_classes, last_epoch=
                        epoch == self.args.epochs - 1, new_class_avg=new_class_averaged,
                                                                     permuted = 'permuted' in self.args.dataset)
                        if self.args.norm_vis and itera == 2:
                            weights = self.model.fc.weight.data if 'cn' not in self.args.method else torch.cat(
                                (self.model.fc.fc1.weight.data, self.model.fc.fc2.weight.data), dim=0)
                            norms = torch.norm(weights, p=2, dim=1, keepdim=True)
                            print("here===========", epoch)
                            self.visualizer.epoch_to_norm[epoch] = {label: norms[label].item() for
                                                                    label in range(self.seen_cls)}
                    print(f"Loss: {train_loss}")

                    if train_loss < min_loss:
                        last_improvement_epoch = epoch
                        min_loss = train_loss
                    if last_improvement_epoch - epoch >= PATIENCE:
                        print("Early stopping of training..")
                        break
                if itera > 0 and 'gem' not in self.args.method and self.args.norm_vis and epoch == self.args.epochs - 1:
                    weights = self.model.fc.weight.data if 'cn' not in self.args.method else torch.cat(
                        (self.model.fc.fc1.weight.data, self.model.fc.fc2.weight.data), dim=0)
                    biases = None if 'cn' in self.args.method else self.model.fc.bias.data.detach().cpu()  # CosineLinearLayer() has no biases, only weights
                    self.visualizer.visualize_new_and_old_weights(weights.detach().cpu(), itera, self.seen_cls,
                                                                  self.dataset.classes_by_groups,
                                                                  biases=biases)
                    if itera == 2:
                        self.visualizer.plot_weight_norms_by_epochs(itera, self.seen_cls)
            test_stats = ""
            if any([x in self.args.method for x in ['ewc', 'mas']]):
                # Source: https://github.com/ElectronicTomato/continue_leanrning_agem/blob/master/agents/regularization.py#L31
                print("Calculating importance score for parameters..")
                task_param = {}
                for n, p in self.params.items():
                    task_param[n] = p.clone().detach()
                importance = self.calculate_importance(train_data, criterion)
                if any([x in self.args.method for x in ['online_ewc', 'mas']]):
                    if len(self.regularization_terms) > 0:
                        self.regularization_terms[1] = {'importance': importance, 'task_param': task_param}
                    elif len(self.regularization_terms) == 0:
                        self.regularization_terms[itera + 1] = {'importance': importance, 'task_param': task_param}

            if itera > 0 and not any([x in self.args.method for x in ['ce', 'ewc', 'gem', 'mas']]):
                # do weight alignment and bias correction after normal training
                self.model.train(False)
                custom_trainer.remove_hooks()
                # naive_acc = self.test(test_data_all, itera, final_test=False)
                if 'wa' in self.args.method:
                    # test_stats += f"Acc. before aligning weights: {naive_acc}"
                    self.scale_weights_and_biases(itera, train_data, num_new_classes)
                # elif self.args.replace_new_logits:
                #     test_stats += f"Acc. before two-step learning: {naive_acc}"
                #     self.model.train(True)
                #     for epoch in range(self.args.epochs):
                #         # balanced training step for two-step learning in ILOS
                #         train_loss = self.normal_training(train_data_balanced, criterion, optimizer, itera)
                #         print(f"---" * 50 + f"\nTwo-step learning epoch: {epoch}, Current Learning Rate : {cur_lr}, Train loss: {train_loss}")
                elif 'bic' in self.args.method:
                    # test_stats += f"Acc. before bias correction: {naive_acc}"
                    bias_optimizer = torch.optim.Adam(self.bias_layers[itera].parameters(), lr=0.001)
                    self.set_gradients_of_bias_layers(
                        bool_val=True)  # unfreeze corresponding bias layer for optimizing params
                    self.set_gradients_of_model_layers(bool_val=False)  # freeze the model's layers
                    min_bias_loss, last_improvement_epoch_bias = 9999, 0
                    for epoch in range(self.args.epochs):
                        bias_loss = self.normal_training(val_data, criterion, bias_optimizer, itera)
                        if bias_loss < min_bias_loss:
                            last_improvement_epoch_bias = epoch
                            min_bias_loss = bias_loss
                        if (last_improvement_epoch_bias - epoch) > PATIENCE:
                            print("======" * 20)
                            print(
                                f"Bias correction loss stopped improving after {last_improvement_epoch_bias} for "
                                f"{PATIENCE} epochs.. stopping training.")
                            print("=====" * 20)
                            break
                    self.set_gradients_of_bias_layers(False)
                    self.set_gradients_of_model_layers(True)
            end_time_train = timer() - start_time_train
            task_times.append(end_time_train)
            if 'gem' in self.args.method:
                start_time_exemp = timer()
                num_samples = self.max_size // (itera + 1)
                num_samples = min(len(train_data.dataset), num_samples)
                # num_samples = int(0.05 * len(train_data.dataset))
                self.build_cache_memory(itera, train_data, num_samples)
                end_time_exemp += timer() - start_time_exemp
            exemp_times.append(end_time_exemp)

            self.model.eval()
            acc = self.test(test_data_all, itera, final_test=True)
            test_stats += f" Final test accuracy on this task: {acc}"
            test_acc.append(acc)
            test_accs.append(max(test_acc))
            test_stats += f"\nTest accuracies over tasks: {test_accs}"
            test_stats += "\n ---------------------------------------------\n\n"
            write_to_report_and_print(self.outfile, test_stats)

        self.base_task_times.append(task_times[0])
        self.old_task_times.append(sum(task_times[1:]) / len(task_times[1:]))
        self.all_task_times.append(sum(task_times) / len(task_times))
        self.exemp_task_times.append(sum(exemp_times) / len(exemp_times))

        if self.args.acc_vis:
            self.visualizer.plot_detailed_accuracies()
            self.visualizer.plot_by_states()
            self.visualizer.plot_by_classes()

        new_line = "---" * 50
        separator = "\n" + new_line
        write_to_report_and_print(self.outfile, "Random sequence of classes: {}".format([self.original_mapping[item]
                                                                                         for each in
                                                                                         self.dataset.classes_by_groups
                                                                                         for item
                                                                                         in each]) + separator * 2)

    def compute_average_vecs(self, train_x, train_y):
        feature_extractor = basic_model.Net(self.input_dim, self.args.total_classes, self.args.dataset)
        feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])
        dict_of_feature_sums = {}
        features = exemplar_selection.get_embeddings(self.input_dim, np.asarray(train_x), feature_extractor)
        for data, label in zip(features, train_y):
            if label not in dict_of_feature_sums:
                dict_of_feature_sums[label] = data.reshape((1, -1))
            else:
                dict_of_feature_sums[label] = np.sum((dict_of_feature_sums[label].reshape((1, -1)), data), axis=0)
        normalized_dict = {label: np.divide(feature, train_y.count(label)) for label, feature in
                           dict_of_feature_sums.items()}
        return normalized_dict

    def scale_weights_and_biases(self, itera, trainloader, num_new_classes):
        if 'wa1' in self.args.method:
            weights_new = self.model.fc.fc2.weight.data if 'cn' in self.args.method else None
            original_weights = self.model.fc.fc1.weight.data if 'cn' in self.args.method else self.model.fc.weight.data
            aligned_new_weights, alpha = self.get_aligned_weights(original_weights, num_new_classes,
                                                                  split_new_class_weights=weights_new)
            with torch.no_grad():
                if 'cn' in self.args.method:
                    self.model.fc.fc2.weight = torch.nn.Parameter(aligned_new_weights)
                else:
                    aligned_weights = torch.cat(
                        (original_weights[:self.seen_cls - num_new_classes], aligned_new_weights), dim=0)
                    self.model.fc.weight = torch.nn.Parameter(aligned_weights)
            if 'cn' not in self.args.method:
                # CosineLinearLayer() does not have biases
                bias_data = self.model.fc.bias.data
                aligned_new_biases, _ = self.get_aligned_weights(bias_data,
                                                                 len(self.dataset.classes_by_groups[itera]),
                                                                 alpha=alpha)
                with torch.no_grad():
                    aligned_biases = torch.cat((bias_data[:self.seen_cls - num_new_classes], aligned_new_biases), dim=0)
                    self.model.fc.bias = torch.nn.Parameter(aligned_biases)
        elif 'wa2' in self.args.method:
            rescale_factor = self.get_rescaling_factor_by_classes(trainloader, itera)
            num_new_classes = len(self.dataset.classes_by_groups[itera])
            new_class_weights = self.model.fc.fc2.weight.data if 'cn' in self.args.method else None
            original_weights = self.model.fc.fc1.weight.data if 'cn' in self.args.method else self.model.fc.weight.data
            rescaled_weights_old, rescaled_weights_new = self.get_rescaled_weights(original_weights,
                                                                                   num_new_classes, rescale_factor,
                                                                                   new_class_weights)
            if 'cn' not in self.args.method:
                rescaled_weights = torch.cat((rescaled_weights_old, rescaled_weights_new), dim=0)
                rescaled_bias_old, rescaled_bias_new = self.get_rescaled_weights(self.model.fc.bias.data,
                                                                                 num_new_classes,
                                                                                 rescale_factor)
                rescaled_biases = torch.cat((rescaled_bias_old, rescaled_bias_new), dim=0)
                with torch.no_grad():
                    self.model.fc.weight = torch.nn.Parameter(rescaled_weights)
                    self.model.fc.bias = torch.nn.Parameter(rescaled_biases)
            elif 'cn' in self.args.method:
                with torch.no_grad():
                    self.model.fc.fc1.weight = torch.nn.Parameter(rescaled_weights_old)
                    self.model.fc.fc2.weight = torch.nn.Parameter(rescaled_weights_new)

    def get_aligned_weights(self, final_layer_weights, num_new_classes, alpha=None, split_new_class_weights=None):
        if 'cn' in self.args.method:
            old_class_weights, new_class_weights = final_layer_weights, split_new_class_weights
        else:
            old_class_weights, new_class_weights = final_layer_weights[:self.seen_cls - num_new_classes], \
                                                   final_layer_weights[self.seen_cls - num_new_classes:]
        old_class_norms, new_class_norms = [[torch.norm(item, p=2, dim=0) for item in each] for each in
                                            (old_class_weights, new_class_weights)]
        if alpha is None:
            alpha = torch.stack(old_class_norms, dim=0).mean() / torch.stack(new_class_norms, dim=0).mean()
        corrected_new_class_weights = alpha * new_class_weights
        return corrected_new_class_weights, alpha

    def get_rescaling_factor_by_classes(self, dataloader, itera):
        new_class_counts = {self.dataset.label_map[each]: 0 for each in self.dataset.classes_by_groups[itera]}
        old_class_count = 0
        for _, target in dataloader:
            target = target.numpy().flatten()
            count_dict = Counter(target)
            for new_class in new_class_counts:
                new_class_counts[new_class] += count_dict[new_class]
            if 0 in count_dict:
                # since all old classes have same no. of exemplars, assign them the count of label '0'
                old_class_count += count_dict[0]
        n_1 = max(new_class_counts.values())
        all_classes = [i for i in range(self.seen_cls)]
        scale_factor_by_classes = {
            label: n_1 / new_class_counts[label] if label in new_class_counts else n_1 / old_class_count for label in
            all_classes}  # scale factor is the imbalance ratio given by n_1 / n_k
        return scale_factor_by_classes

    def get_rescaled_weights(self, final_layer_weights, num_new_classes, rescaling_factor_dict,
                             split_new_class_weights=None):
        if 'cn' in self.args.method:
            # SplitCosineLayer() has separate heads for old and new classes
            old_class_weights, new_class_weights = final_layer_weights, split_new_class_weights
        else:
            old_class_weights, new_class_weights = final_layer_weights[:self.seen_cls - num_new_classes], \
                                                   final_layer_weights[self.seen_cls - num_new_classes: self.seen_cls]
        new_classes = [i for i in range(self.seen_cls - num_new_classes, self.seen_cls)]
        rescaled_old_class_weights = torch.stack(
            [w * (rescaling_factor_dict[idx] ** self.args.wa2_gamma) for idx, w in enumerate(old_class_weights)])
        rescaled_new_class_weights = torch.stack(
            [w * (rescaling_factor_dict[label] ** self.args.wa2_gamma) for label, w in
             zip(new_classes, new_class_weights)])
        return rescaled_old_class_weights, rescaled_new_class_weights

    def bias_forward(self, input):
        input_groups = []
        for idx, classes in enumerate(self.dataset.classes_by_groups):
            temp_tensor = torch.Tensor().to(self.device)
            for each in classes:
                each = self.dataset.label_map[each]
                temp_tensor = torch.cat([temp_tensor, input[:, (int)(each):(int)(each + 1)]], dim=1)
            input_groups.append(temp_tensor)
        output_by_groups = [self.bias_layers[idx](item) for idx, item in enumerate(input_groups)]
        output_by_groups = torch.cat(output_by_groups, dim=1)
        return output_by_groups

    def grad_to_vector(self):
        """
        Source: https://github.com/ElectronicTomato/continue_leanrning_agem/blob/master/agents/exp_replay.py#L121
        :return:
        """
        vec = []
        for n, p in self.params.items():
            if p.grad is not None:
                vec.append(p.grad.view(-1))
            else:
                vec.append(p.data.clone().fill_(0.0).view(-1))
        return torch.cat(vec)

    def vector_to_grad(self, vec):
        """
        Source: https://github.com/ElectronicTomato/continue_leanrning_agem/blob/master/agents/exp_replay.py#L131
        :param vec:
        :return:
        """
        # Overwrite current param.grad by slicing the values in vec (flatten grad)
        pointer = 0
        for n, p in self.params.items():
            # The length of the parameter
            num_param = p.numel()
            if p.grad is not None:
                # Slice the vector, reshape it, and replace the old data of the grad
                p.grad.copy_(vec[pointer:pointer + num_param].view_as(p))
                # Part of the network might has no grad, ignore those terms
            # Increment the pointer
            pointer += num_param

    def get_grad(self, current_grad, previous_avg_grad):
        """
        Eqn (11) from "EFFICIENT LIFELONG LEARNING WITH A-GEM"
        Source: https://github.com/ElectronicTomato/continue_leanrning_agem/blob/master/agents/exp_replay.py#L144
        Official tensorflow implementation: https://github.com/facebookresearch/agem/blob/master/model/model.py#L1163
        :param current_grad: gradeints for current task instances
        :param previous_avg_grad: averaged gradients for all previous task instances
        :return:
        """
        dotp = (current_grad * previous_avg_grad).sum()
        ref_mag = (previous_avg_grad * previous_avg_grad).sum()
        new_grad = current_grad - ((dotp / ref_mag) * previous_avg_grad)
        new_grad = new_grad.to(self.device)
        return new_grad

    def gem_training(self, data_loader, criterion, optimizer, itera):
        """
        Function to do training for GEM and AGEM methods.
        Implementation adjusted from https://github.com/ElectronicTomato/continue_leanrning_agem/blob/master/agents/exp_replay.py#L379
        :param data_loader: train loader object
        :param criterion: cross-entropy loss
        :param optimizer: SGD optimizer
        :param itera: incremental batch no.
        :return: loss averaged over data
        """
        print("Training ... ")
        losses = []
        for i, (data, label) in enumerate(tqdm(data_loader)):
            loss_report = "\n"
            # compute gradient on previous tasks
            for t, mem in self.task_memory.items():
                self.model.zero_grad()
                mem_out = self.model(self.task_mem_cache[t]['data'])
                mem_loss = criterion(mem_out, self.task_mem_cache[t]['target'])
                loss_report += f" Old task ID: {t}, {'AGEM' if self.args.method == 'agem' else 'GEM'} loss: {mem_loss}\n"
                mem_loss.backward()
                self.task_grads[t] = self.grad_to_vector()

            data, label = Variable(data), Variable(label)
            data = data.to(self.device)
            # label = label.type(torch.LongTensor)
            label = label.view(-1).to(self.device)
            p = self.model(data.float())
            loss = criterion(p[:, :self.seen_cls], label)
            loss_report += f"CE loss: {loss}"
            optimizer.zero_grad()
            loss.backward(retain_graph=itera == 0)

            # check if gradient violates constraints
            current_grad_vec = self.grad_to_vector()
            # reference grad should be average gradient of all previous tasks
            ref_grad_vec = torch.stack(list(self.task_grads.values()))
            if self.args.method == 'agem':
                # if agem, calculate the average over all previous tasks
                ref_grad_vec = torch.sum(ref_grad_vec, dim=0) / ref_grad_vec.shape[0]
                assert current_grad_vec.shape == ref_grad_vec.shape
            dotp = current_grad_vec * ref_grad_vec
            dotp = dotp.sum(dim=1) if self.args.method == 'gem' else dotp.sum()
            if (dotp < 0).sum() != 0:
                if self.args.method == 'agem':
                    new_grad = self.get_grad(current_grad_vec, ref_grad_vec)
                elif self.args.method == 'gem':
                    new_grad = project2cone2(current_grad_vec, ref_grad_vec, self.device)
                self.vector_to_grad(new_grad)

            optimizer.step()
            print(loss_report)
            losses.append(loss.detach().item())
        return sum(losses) / len(data_loader.dataset)

    def calculate_importance(self, dataloader, criterion):
        """
        Method to update the diagonal fisher information.
        Source: https://github.com/ElectronicTomato/continue_leanrning_agem/blob/master/agents/regularization.py#L89
        :param dataloader: train loader object
        :param criterion: cross entropy
        :return:
        """
        if self.args.method in ['online_ewc', 'mas'] and len(self.regularization_terms) > 0:
            # use only one importance matrix across all tasks
            importance = self.regularization_terms[1]['importance']
        else:
            # compute a new importance matrix for the current task
            importance = {}
            for n, p in self.params.items():
                importance[n] = p.clone().detach().fill_(0)  # zero initialized

        if 'ewc' in self.args.method and self.n_fisher_sample is not None:
            # estimate fisher matrix using a subset of data; saves computation time
            n_sample = min(self.n_fisher_sample, len(dataloader.dataset))
            rand_ind = random.sample(list(range(len(dataloader.dataset))), n_sample)
            subdata = torch.utils.data.Subset(dataloader.dataset, rand_ind)
            dataloader = torch.utils.data.DataLoader(subdata, shuffle=True, batch_size=1)

        self.model.eval()
        for i, (input, target) in enumerate(dataloader):
            input, target = Variable(input), Variable(target)
            input = input.to(self.device)
            target = target.view(-1).to(self.device)
            preds = self.model(input.float())[:, :self.seen_cls]
            if 'ewc' in self.args.method:
                ind = preds.max(1)[1].flatten()
                loss = criterion(preds, ind)
            elif 'mas' in self.args.method:
                preds.pow_(2)
                loss = preds.mean()
            print(f"Fisher estimation loss: {loss.item()}")
            self.model.zero_grad()
            loss.backward()
            for n, p in importance.items():
                if self.params[n].grad is not None:
                    if 'ewc' in self.args.method:
                        importance[n] += ((self.params[n].grad ** 2) * len(input) / len(dataloader))
                    elif 'mas' in self.args.method:
                        importance[n] += (self.params[n].grad.abs() / len(dataloader))
        return importance

    def normal_training(self, data_loader, criterion, optimizer, itera):
        """
        Method for normal training with only cross entropy (for first offline batch)
        :param data_loader: train loader object
        :param criterion: cross entropy loss
        :param optimizer: SGD
        :param itera: incremental batch no.
        :return: loss averaged over the data
        """
        print("Training ... ")
        losses = []
        for i, (data, label) in enumerate(tqdm(data_loader)):
            data, label = Variable(data), Variable(label)
            if self.args.dataset == 'cifar100':
                data = data.reshape(128, -1)
            data = data.to(self.device)
            # label = label.type(torch.LongTensor)
            label = label.view(-1).to(self.device)
            p = self.model(data.float())
            if 'bic' in self.args.method and itera > 0:
                # this condition holds when we need to train the bias layers for incremental batches from itera = 1...N
                p = self.bias_forward(p)
                gamma_l2_loss = self.bias_layers[itera].get_gamma().norm(2)
                loss = criterion(p, label) + gamma_l2_loss * 0.1
                loss_stats = f"BiC loss: {loss}"
            else:
                loss = criterion(p, label)
                loss_stats = f"CE loss: {loss}"
                if any([x in self.args.method for x in ['mas', 'online_ewc']]) and len(self.regularization_terms) > 0:
                    reg_loss = 0
                    for i, reg_term in self.regularization_terms.items():
                        task_reg_loss = 0
                        importance = reg_term['importance']
                        task_param = reg_term['task_param']
                        for n, p in self.params.items():
                            task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()
                        reg_loss += task_reg_loss
                    loss += reg_loss * self.args.reg_coef if self.args.reg_coef > 0 else reg_loss
            print(loss_stats)
            optimizer.zero_grad()
            loss.backward(retain_graph=itera == 0)
            optimizer.step()
            losses.append(loss.item())
        return sum(losses) / len(data_loader.dataset)

    def test(self, testdata, itera, final_test=False):
        # test set is always 0.2 * the data seen so far
        k = 3 if itera > 0 else 2
        y_pred, y_true = [], []
        top1_acc, topk_acc = [], []
        if self.args.tsne_vis and itera > 0:
            tsne_features, tsne_labels = np.empty(shape=[0, self.model.fc.in_features]), []
            register_test_hook = self.model.fc.register_forward_hook(customized_distill_trainer.get_cur_features)

        for i, (data, label) in enumerate(testdata):
            data, label = Variable(data), Variable(label)
            data = data.to(self.device)
            # label = label.type(torch.LongTensor)
            label = label.view(-1).to(self.device)
            with torch.no_grad():
                p = self.model(data.float())
                if 'bic' in self.args.method:
                    p = self.bias_forward(p)
            pred = p[:, :self.seen_cls].argmax(dim=-1)
            prec1, preck = self.topk_accuracy(p.data, label, topk=(1, k))
            top1_acc.append(prec1.item())
            topk_acc.append(preck.item())
            labels_list = label.cpu().tolist()
            preds_list = pred.cpu().tolist()
            y_true += [self.original_mapping[self.label_map[each]] for each in labels_list]
            y_pred += [self.original_mapping[self.label_map[each]] for each in preds_list]
            if self.args.tsne_vis and itera > 0:
                tsne_features = np.vstack((tsne_features, customized_distill_trainer.cur_features.detach().cpu().numpy()))
                tsne_labels += labels_list

        if self.args.tsne_vis and itera > 0:
            register_test_hook.remove()
            print(f"Generating test t-SNE for incremental task: {itera}")
            self.visualizer.plot_tsne(tsne_features, tsne_labels, itera=itera, test=True)

        report = classification_report(y_pred=y_pred, y_true=y_true, output_dict=True)
        mispred_results = self.analyze_mispredictions(y_true, y_pred, itera)

        df = pd.DataFrame(report).transpose()
        print(df)
        top1_mean, topk_mean = [sum(each) / len(each) for each in (top1_acc, topk_acc)]
        result = f"Test acc: top1 = {top1_mean}, top{k} = {topk_mean} Test set size: {len(y_true)}"
        result += mispred_results
        print(result)

        if itera > 0:
            self.compute_prediction_distance_by_states(y_pred, y_true, itera, final_test=final_test)

        if final_test:
            self.compute_prediction_distance_by_classes(y_pred, y_true, itera)
            df.to_csv(self.outfile, sep='\t', mode='a')
            if itera > 0:
                detailed_acc_ = self.evaluate_detailed_accuracies(y_pred, y_true, itera)
                result += detailed_acc_
            write_to_report_and_print(self.outfile, result)
            if self.args.corr_vis:
                conf_mat_dataframe = self.compute_confusion_matrix(y_true, y_pred, itera)
                # self.visualizer.plot_diagonal_correlation_matrix(conf_mat_dataframe, itera)
                self.visualizer.plot_confusion_matrix(conf_mat_dataframe, itera)
            if itera == self.dataset.num_tasks - 1 and self.args.average_over != 'na':
                key_ = self.train_percent if self.args.average_over == 'tp' else self.holdout_size
                self.list_of_classification_reports[key_].append(report)
                # self.detailed_accuracies[key_]['micro'].append(self.visualizer.detailed_micro_scores)
                # self.detailed_accuracies[key_]['macro'].append(self.visualizer.detailed_macro_scores)
        return top1_mean

    def topk_accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k
        Source: https://github.com/EdenBelouadah/class-incremental-learning/blob/master/il2m/codes/utils/Utils.py#L11
        """
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def compute_confusion_matrix(self, y_true, y_pred, itera):
        """
        Confusion matrix of true and predicted classes
        :param y_true: ground truth
        :param y_pred: predictions
        :param itera: no. of incremental batch
        :return: pandas dataframe containing confusion matrix
        """
        classes_seen = [self.original_mapping[item] for each in self.dataset.classes_by_groups[:itera + 1]
                        for item in each]
        classes_seen.sort()
        conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=classes_seen)
        df_conf_mat = pd.DataFrame(conf_mat, columns=classes_seen, index=classes_seen)
        return df_conf_mat

    def analyze_mispredictions(self, y_true, y_pred, itera):
        """
        Compares the confusion between old and new classes
        :param y_true: ground truth
        :param y_pred: predictions
        :param itera: incrmental batch
        :return: string detailing confusion for old classes: e_o, old with new / other old classes: e_o_n / e_o_o,
         and new with other new classes: e_n
        """
        analyzer = prediction_analyzer.PredictionAnalysis(y_true, y_pred, self.dataset, self.original_mapping,
                                                          self.label_map)
        mispredicted_stats = analyzer.analyze_misclassified_instances(batch_num=itera)
        if itera > 0:
            misclassified_new_classes, total_new_classes, misclassified_old_classes, total_old_classes, old_misclassified_new, old_misclassified_old = mispredicted_stats
            mispred = f"\n e(n): {misclassified_new_classes}, total_new: {total_new_classes} e(o): {misclassified_old_classes}," \
                      f" total_old: {total_old_classes}, e(o,n): {old_misclassified_new}, e(o,o): {old_misclassified_old}"
            if itera == self.dataset.num_tasks - 1 and self.args.average_over != 'na':
                key_ = self.holdout_size if self.args.average_over == 'holdout' else self.train_percent
                self.error_stats[key_]['e_o'].append(misclassified_old_classes)
                self.error_stats[key_]['e_n'].append(misclassified_new_classes)
                self.error_stats[key_]['e_o_n'].append(old_misclassified_new)
                self.error_stats[key_]['e_o_o'].append(old_misclassified_old)

        else:
            misclassified_new_classes, total_new_classes = mispredicted_stats
            mispred = f"\n e(n): {misclassified_new_classes}, total_new: {total_new_classes}"
        return mispred

    def evaluate_detailed_accuracies(self, y_pred, y_true, itera):
        """
        Computes averaged accuracies over base, old and new classes after training is over
        :param y_pred: predictions
        :param y_true: ground truth
        :param itera: no. of incremental batch
        :return: string detailing averaged base, old and new class scores
        """
        new_classes = [self.original_mapping[each] for each in self.dataset.classes_by_groups[itera]]
        old_classes = [self.original_mapping[item] for each in self.dataset.classes_by_groups[:itera]
                       for item in each]
        base_classes = [self.original_mapping[each] for each in self.dataset.classes_by_groups[0]]
        old_indices, base_indices, new_indices = [[idx for idx, val in enumerate(y_true) if val in each] for each in
                                                  [old_classes, base_classes, new_classes]]

        for indices in [base_indices, old_indices, new_indices]:
            # base, old and new classes
            filtered_list = [(pred_val, true_val) for idx, (pred_val, true_val) in enumerate(zip(y_pred, y_true)) if
                             idx in indices]
            _y_pred = [each[0] for each in filtered_list]
            _y_true = [each[1] for each in filtered_list]
            self.visualizer.detailed_micro_scores[itera - 1] += (f1_score(_y_true, _y_pred, average='micro') * 100,)
            self.visualizer.detailed_macro_scores[itera - 1] += (f1_score(_y_true, _y_pred, average='macro') * 100,)

        self.visualizer.detailed_micro_scores[itera - 1] += (
        f1_score(y_true, y_pred, average='micro') * 100,)  # all classes
        self.visualizer.detailed_macro_scores[itera - 1] += (f1_score(y_true, y_pred, average='macro') * 100,)

        base_class_micro, base_class_macro = self.visualizer.detailed_micro_scores[itera - 1][0], \
                                             self.visualizer.detailed_macro_scores[itera - 1][0]
        old_class_micro, old_class_macro = self.visualizer.detailed_micro_scores[itera - 1][1], \
                                           self.visualizer.detailed_macro_scores[itera - 1][1]
        new_class_micro, new_class_macro = self.visualizer.detailed_micro_scores[itera - 1][2], \
                                           self.visualizer.detailed_macro_scores[itera - 1][2]
        all_class_micro, all_class_macro = self.visualizer.detailed_micro_scores[itera - 1][3], \
                                           self.visualizer.detailed_macro_scores[itera - 1][3]

        index_ = self.holdout_size if self.args.average_over == 'holdout' else self.train_percent
        self.detailed_accuracies[index_]['micro'][itera].append(
            [base_class_micro, old_class_micro, new_class_micro, all_class_micro])
        self.detailed_accuracies[index_]['macro'][itera].append(
            [base_class_macro, old_class_macro, new_class_macro, all_class_macro])

        result = f"\n \t\t\t F1-micro \t\t F1-macro \n"
        result += f"Base classes: {base_class_micro} \t {base_class_macro}\n"
        result += f"Old classes: {old_class_micro} \t {old_class_macro}\n"
        result += f"New classes: {new_class_micro} \t {new_class_macro}\n"
        result += f"All classes: {all_class_micro} \t {all_class_macro}\n"

        return result

    @staticmethod
    def get_mean_scores(y_pred, y_true):
        """
        Computes mean scores by each class label
        :param y_pred: ground truth
        :param y_true: predictions
        :return:  dict of class label to mean scores
        """
        all_classes = set(y_true)
        class_to_scores = {_class: 0.0 for _class in all_classes}
        for each_class in all_classes:
            total_count = y_true.count(each_class)
            true_count = len([idx for idx, (first, second) in enumerate(zip(y_true, y_pred)) if
                              first == second and first == each_class])
            class_to_scores[each_class] = true_count / total_count
        return class_to_scores

    def compute_prediction_distance_by_states(self, y_pred, y_true, itera, final_test=False):
        """
        Calculated accuracy by incremental states for old and new class samples
        :param y_pred: predictions
        :param y_true: ground truth
        :param itera: incremental batch no.
        :param final_test: flag indicating whether to calculate before or after weight alignment / bias correction
        :return: dictionary of incremental state to accuracy scores of old and new classes
        """
        current_classes = [self.original_mapping[each] for each in
                           self.dataset.classes_by_groups[itera]]
        previous_classes = [self.original_mapping[item] for each in
                            self.dataset.classes_by_groups[:itera] for item in each]

        class_to_scores = self.get_mean_scores(y_pred, y_true)
        previous_class_mean = sum([class_to_scores[key] for key, val in class_to_scores.items() if key in
                                   previous_classes]) / float(len(previous_classes))
        current_class_mean = sum([class_to_scores[key] for key, val in class_to_scores.items() if key in
                                  current_classes]) / float(len(current_classes))
        if 'bic' in self.args.method or 'wa' in self.args.method:
            if not final_test:
                self.visualizer.incr_state_to_scores[itera] = (
                    previous_class_mean, current_class_mean)  # scores without bias removal
            else:
                self.visualizer.incr_state_to_scores[itera] += (
                    previous_class_mean, current_class_mean)  # append scores after bias removal
        else:
            self.visualizer.incr_state_to_scores[itera] = (previous_class_mean, current_class_mean)

    def compute_prediction_distance_by_classes(self, y_pred, y_true, itera):
        """
        Calculated accuracy by classes when they were new and averaged over all succeeding states when they were old
        :param y_pred: predictions
        :param y_true: ground truth
        :param itera: incremental batch no.
        :return: dictionary of new and previous classes to accuracy scores
        """
        current_classes = [self.original_mapping[each] for each in
                           self.dataset.classes_by_groups[itera]]
        previous_classes = [self.original_mapping[item] for each in
                            self.dataset.classes_by_groups[:itera] for item in each]
        class_to_scores = self.get_mean_scores(y_pred, y_true)
        self.visualizer.current_class_scores.update(
            {key: val for key, val in class_to_scores.items() if key in current_classes})
        for key, val in class_to_scores.items():
            if key in previous_classes:
                if key not in self.visualizer.previous_class_scores:
                    self.visualizer.previous_class_scores[key] = [val]
                else:
                    self.visualizer.previous_class_scores[key].append(val)
