import math
from collections import Counter

import numpy as np
import torch.nn as nn
from pydpp.dpp import DPP
from sklearn.neighbors import NearestNeighbors
from torch import from_numpy, no_grad
from torch.autograd import Variable

import models.basic_model as models
from .exemplar_strategies import fwsr, class_boundary, herding, kmeans
from .visualisations import exemplar_visualizer


def get_embeddings(input_dim, instance, feature_extractor):
    if len(instance.shape) > 1:
        # implies an array of features: (126, 243)
        features = []
        for data in instance:
            with no_grad():
                x = Variable(from_numpy(data))
                feature = feature_extractor(x.view(-1, input_dim).float()).data.numpy()
                feature = feature / np.linalg.norm(feature)
                features.append(feature[0])

    elif len(instance.shape) == 1:
        # implies a single feature: (243, )
        with no_grad():
            x = Variable(from_numpy(instance))
            feature = feature_extractor(x.view(-1, input_dim).float())[0].data.numpy()
            features = feature / np.linalg.norm(feature)

    return features


class Exemplar:
    def __init__(self, dataname, max_size, total_classes, input_dim, reversed_label_map, reversed_orig_map, outfile,
                 device):
        self.val = {}
        self.train = {}
        self.cur_cls = 0
        self.dataname = dataname
        self.max_size = max_size
        self.total_classes = total_classes
        self.virtual_mapping = reversed_label_map
        self.original_mapping = reversed_orig_map
        self.input_dim = input_dim
        self.outfile = outfile
        self.train_to_indices = {}  # used for visualizing selected exemplars
        self.task_mem_cache = {}
        self.device = device

    @staticmethod
    def get_dict_by_class(features, labels):
        classwise_dict_of_features = {}
        for idx, label in enumerate(labels):
            if label not in classwise_dict_of_features:
                classwise_dict_of_features[label] = [features[idx]]
            else:
                classwise_dict_of_features[label].append(features[idx])
        return classwise_dict_of_features

    def extract_features_of_class(self, list_of_instances, feature_extractor):
        features = []
        for each in list_of_instances:
            feature = get_embeddings(self.input_dim, each, feature_extractor)
            features.append(feature)
        return features

    @staticmethod
    def DPPsample(X, k, kernel, sigma):
        default_k = 2
        nk = k
        indices = []
        dpp = DPP(X)
        if (kernel == 'cos-sim'):
            dpp.compute_kernel(kernel_type=kernel)
        else:
            dpp.compute_kernel(kernel_type='rbf', sigma=sigma)
        if (k < default_k):
            idx = dpp.sample_k(k)
            indices = indices + idx.tolist()
        else:
            while (nk > 0):
                if (nk > default_k):
                    idx = dpp.sample_k(default_k)
                else:
                    idx = dpp.sample_k(nk)
                indices = indices + idx.tolist()
                #             X=X.delete(.drop(idx)
                np.delete(X, idx, 0)
                nk = nk - default_k
        return indices

    def extract_features(self, train_dict, val_dict=None):
        feature_extractor = models.Net(self.input_dim, self.total_classes, self.dataname)
        feature_extractor.fc = nn.Linear(feature_extractor.fc.in_features, self.input_dim)

        train_dict_features = {key: self.extract_features_of_class(val, feature_extractor) for key, val in
                               train_dict.items()}
        val_dict_features = {key: self.extract_features_of_class(val, feature_extractor) for key, val in
                             val_dict.items()} if val_dict is not None else None
        return train_dict_features, val_dict_features

    def icarl_update(self, train_dict, val_dict=None):
        # discard fraction of previous exemplars so that the old classes now have reduced no. of instances
        for old_cl, value in self.train.items():
            value = np.array(value)
            self.train[old_cl] = value[:self.train_store_dict[old_cl]]

        if val_dict is not None:
            for old_cl, value in self.val.items():
                value = np.array(value)
                self.val[old_cl] = value[:self.val_store_dict[old_cl]]

        train_dict_features, val_dict_features = self.extract_features(train_dict, val_dict)
        for label, features in train_dict_features.items():
            # perform herding based sorted indices selection for new classes
            nearest_indices = herding.herding_selection(features, self.train_store_dict[label])
            self.train[label] = np.array(train_dict[label])[nearest_indices]
            if val_dict is not None:
                nearest_indices_val = herding.herding_selection(val_dict_features[label], self.val_store_dict[label])
                self.val[label] = np.array(val_dict[label])[nearest_indices_val]

    def random_update(self, train_dict, val_dict=None):
        for old_cl, value in self.train.items():
            value = np.array(value)
            self.train[old_cl] = value[np.random.choice(len(value), self.train_store_dict[old_cl], replace=False)]

        if val_dict is not None:
            for old_cl, value in self.val.items():
                value = np.array(value)
                self.val[old_cl] = value[np.random.choice(len(value), self.val_store_dict[old_cl], replace=False)]

        for new_cl, features in train_dict.items():
            value = np.array(features)
            print(new_cl, len(value), self.train_store_dict[new_cl])
            random_indices = np.random.choice(len(value), self.train_store_dict[new_cl], replace=False)
            self.train[new_cl] = value[random_indices]
            self.train_to_indices[new_cl] = random_indices
        if val_dict is not None:
            for new_cl, features in val_dict.items():
                value = np.array(features)
                print(new_cl, len(value), self.val_store_dict[new_cl])
                self.val[new_cl] = value[np.random.choice(len(value), self.val_store_dict[new_cl], replace=False)]

    def clustered_update(self, sample_algo, train_dict, val_dict=None):
        for old_cl, value in self.train.items():
            value = np.array(value)
            self.train[old_cl] = value[:self.train_store_dict[old_cl]]

        for new_cl, X in train_dict.items():
            X = np.array(X)
            filter_indices = self.DPPsample(X, self.train_store_dict[new_cl], 'cos-sim', 0) if sample_algo == 'dpp' else \
                kmeans.kmeans_sample(X, self.train_store_dict[new_cl]) if sample_algo == 'kmeans' else None
            self.train[new_cl] = X[filter_indices]
            self.train_to_indices[new_cl] = filter_indices

        if val_dict is not None:
            for old_cl, value in self.val.items():
                value = np.array(value)
                self.val[old_cl] = value[:self.val_store_dict[old_cl]]

            for new_cl, X in val_dict.items():
                X = np.array(X)
                filter_indices = self.DPPsample(X, self.val_store_dict[new_cl], 'cos-sim',
                                                0) if sample_algo == 'dpp' else \
                    kmeans.kmeans_sample(X, self.val_store_dict[new_cl]) if sample_algo == 'kmeans' else None
                self.val[new_cl] = X[filter_indices]

    def perform_fwsr_update(self, train_dict, val_dict=None):
        for old_cl, value in self.train.items():
            value = np.array(value)
            self.train[old_cl] = value[:self.train_store_dict[old_cl]]
        train_dict_features, val_dict_features = self.extract_features(train_dict, val_dict)
        for label, features in train_dict_features.items():
            A = np.array(features).transpose()
            K = A.transpose() @ A
            indices = fwsr.FWSR_identify_exemplars(A=A, K=K, max_iterations=2000,
                                                   num_exemp=self.train_store_dict[label],
                                                   beta=self.train_store_dict[label] * 4)
            if len(indices) < self.train_store_dict[label]:
                indices_not_chosen = np.setdiff1d(np.array([i for i in range(len(features))]), indices)
                random_indices = np.random.choice(indices_not_chosen, self.train_store_dict[label] - len(indices),
                                                  replace=False)
                indices = np.hstack((indices, random_indices))
            self.train[label] = np.array(train_dict[label])[[indices[:self.train_store_dict[label]]]]
            self.train_to_indices[label] = indices

        if val_dict is not None:
            for old_cl, value in self.val.items():
                value = np.array(value)
                self.val[old_cl] = value[:self.val_store_dict[old_cl]]

            for label, features in val_dict_features.items():
                A = np.array(features).transpose()
                K = A.transpose() @ A
                indices = fwsr.FWSR_identify_exemplars(A=A, K=K, max_iterations=2000,
                                                       num_exemp=self.val_store_dict[label],
                                                       beta=self.val_store_dict[label] * 4)
                if len(indices) < self.val_store_dict[label]:
                    indices_not_chosen = np.setdiff1d(np.array([i for i in range(len(features))]), indices)
                    random_indices = np.random.choice(indices_not_chosen, self.val_store_dict[label] - len(indices),
                                                      replace=False)
                    indices = np.hstack((indices, random_indices))
                self.val[label] = np.array(val_dict[label])[[indices[:self.val_store_dict[label]]]]

    @staticmethod
    def get_sampling_probabilities(class_to_consider, label_to_features_dict, train_store_num,
                                   list_of_sums, dict_of_sums, nbrs, k, old_cls_update=False):
        # compute gamma+ and gamma- for each majority example - c+ = classes from old tasks, c- = from new task
        sampling_probabilites = []
        calibration_factor = train_store_num / len(label_to_features_dict[class_to_consider])
        for data_point in label_to_features_dict[class_to_consider]:
            distances, indices = nbrs.kneighbors(data_point.reshape(1, -1))
            gamma_plus, gamma_minus = 0., 0.
            for index in indices[0][1:]:  # since first index always contains the element itself
                # get sum value that is closest to this index
                nearest_sum = min(list_of_sums, key=lambda x: (x - index) if x >= index else max(list_of_sums))
                nearest_class = dict_of_sums[nearest_sum]
                if nearest_class == class_to_consider:
                    gamma_minus += 1
                else:  # also consider only this class
                    gamma_plus += 1
            gamma_plus, gamma_minus = gamma_plus / k, gamma_minus / k
            if not old_cls_update:
                # since new classes have relatively high number of samples, gamma_minus dominates and needs to be adjusted
                gamma_minus_adjusted = calibration_factor * gamma_minus
                try:
                    sensitivity = (gamma_minus / (gamma_plus + gamma_minus)) - (gamma_minus_adjusted / (gamma_plus +
                                                                                                        gamma_minus_adjusted))
                except ZeroDivisionError:
                    sensitivity = 0.
            else:
                try:
                    sensitivity = gamma_minus / (gamma_plus + gamma_minus)  # no need for gamma_minus calibration here
                except ZeroDivisionError:
                    sensitivity = 0.
            weight = sensitivity + 1 / len(label_to_features_dict[class_to_consider])
            # print(f"Gamma plus: {gamma_plus}, gamma minus: {gamma_minus}, sensitivity: {sensitivity}, weight: {weight}")
            sampling_probabilites.append(weight)
        prob_sum = sum(sampling_probabilites)
        sampling_probabilites = [prob / prob_sum for prob in sampling_probabilites]
        return sampling_probabilites

    def uus_sensitivity_update(self, label_to_features_dict, classes_to_update, old_cls_update=False, train=True):
        """
        Function to update train and validation exemplars of the classes supplied.
        :param label_to_features_dict:
        :param val_dict:
        :param classes_to_update: classes whose train and validation exemplars are to be updated.
        :param train_store_num:
        :param val_store_num:
        :param old_cls_update:
        :return:
        """

        # update train and val dicts with currently stored exemplars - gives us the whole data
        if train:
            label_to_features_dict.update(self.train)
        else:
            label_to_features_dict.update(self.val)

        """ Undersampling method taken from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8949290 """
        """ Nearest neighbours sklearn; 
        Source: https://scikit-learn.org/stable/modules/neighbors.html """
        new_classes_len = [len(label_to_features_dict[label]) for label in classes_to_update]
        total_new_class_samples = 0
        total_new_class_samples += sum(new_classes_len)

        dataset_array = np.vstack(list(label_to_features_dict.values()))
        length_by_classes = [len(each) for each in list(label_to_features_dict.values())]
        list_of_sums = [sum(length_by_classes[:idx + 1]) for idx in range(len(length_by_classes))]
        dict_of_sums = {sum_val: key for sum_val, key in zip(list_of_sums, label_to_features_dict.keys())}

        k = math.ceil(min(new_classes_len) / 1.25)
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(dataset_array)

        for majority_class in classes_to_update:
            desired_holdout_size = self.train_store_dict[majority_class] if train else self.val_store_dict[
                majority_class]
            sampling_prob = self.get_sampling_probabilities(majority_class, label_to_features_dict,
                                                            desired_holdout_size,
                                                            list_of_sums, dict_of_sums, nbrs, k, old_cls_update)
            random_indices = np.random.choice(len(label_to_features_dict[majority_class]), desired_holdout_size,
                                              replace=False, p=sampling_prob)
            if train:
                self.train[majority_class] = np.array(label_to_features_dict[majority_class])[random_indices]
                self.train_to_indices[majority_class] = random_indices
            else:
                self.val[majority_class] = np.array(label_to_features_dict[majority_class])[random_indices]

    def sensitivity_update(self, train_dict, val_dict=None):
        classes_to_consider = list(self.train.keys())  # update for old classes
        self.uus_sensitivity_update(train_dict.copy(), classes_to_consider, old_cls_update=True, train=True)
        if val_dict is not None:
            self.uus_sensitivity_update(val_dict.copy(), classes_to_consider, old_cls_update=True, train=False)
        classes_to_consider = set(train_dict) - set(self.train)  # update for new classes
        self.uus_sensitivity_update(train_dict.copy(), classes_to_consider, train=True)
        if val_dict is not None:
            self.uus_sensitivity_update(val_dict.copy(), classes_to_consider, train=False)

    @staticmethod
    def get_normalized_mean(features):
        features = np.array(features)
        class_mean = np.mean(features, axis=0)
        return class_mean / np.linalg.norm(class_mean)

    def boundary_update(self, train_dict, val_dict=None):
        for old_cl, value in self.train.items():
            value = np.array(value)
            self.train[old_cl] = value[:self.train_store_dict[old_cl]]

        train_dict_features, val_dict_features = self.extract_features(train_dict, val_dict)
        train_dict_mean = {key: self.get_normalized_mean(val) for key, val in train_dict_features.items()}
        updated_exemplar_dict_train = class_boundary.get_new_exemplars(train_dict, train_dict_features, train_dict_mean,
                                                                       exemp_size_dict=self.train_store_dict)
        for label, exemplars in updated_exemplar_dict_train.items():
            self.train[label] = exemplars

        if val_dict is not None:
            for old_cl, value in self.val.items():
                value = np.array(value)
                self.val[old_cl] = value[:self.val_store_dict[old_cl]]

            val_dict_mean = {key: self.get_normalized_mean(val) for key, val in val_dict_features.items()}
            updated_exemplar_dict_val = class_boundary.get_new_exemplars(val_dict, val_dict_features, val_dict_mean,
                                                                         exemp_size_dict=self.val_store_dict)
            for label, exemplars in updated_exemplar_dict_val.items():
                self.val[label] = exemplars

    def get_holdout_size_by_labels(self, count_of_labels, store_num, val=False):
        sorted_count_dict = sorted(count_of_labels, key=count_of_labels.get)
        dict_of_store_size = {}
        for label in sorted_count_dict:
            # compute sizes for new classes
            true_size = min(store_num, count_of_labels[label])
            dict_of_store_size[label] = true_size
        for old_cl in self.train:
            # sizes for old classes
            if val:
                dict_of_store_size[old_cl] = min(store_num, len(self.val[old_cl]))
            else:
                dict_of_store_size[old_cl] = min(store_num, len(self.train[old_cl]))
        return dict_of_store_size

    def update(self, strategy, cls_num, task_num, vis=False, train=None, val=None):
        train_x, train_y = train
        assert  self.cur_cls == 0 or self.cur_cls % len(list(self.train.keys())) == 0 # for permuted mnist, seen cls = always multiple of total classes
        self.cur_cls += cls_num
        samples_per_class = self.max_size / self.cur_cls if self.cur_cls != 0 else self.max_size
        train_percent = 0.9 if val is not None else 1.0
        val_percent = 1 - train_percent
        val_store_num = math.ceil(
            samples_per_class * val_percent)  # guarantees minimum of one val sample per class
        train_store_num = math.ceil(
            samples_per_class) - val_store_num

        train_y_counts = Counter(train_y)
        self.train_store_dict = self.get_holdout_size_by_labels(train_y_counts, train_store_num)
        train_dict, val_dict = self.get_dict_by_class(train_x, train_y), None

        new_labels = set(train_y)
        print(new_labels, self.original_mapping, self.virtual_mapping)
        mapped_new_classes = [self.original_mapping[self.virtual_mapping[each]] for each in new_labels]
        mapped_old_classes = [self.original_mapping[self.virtual_mapping[each]] for each in self.train.keys()]

        exemplar_details = "\nNew classes: {}, Old classes: {}\nUpdated memory size for each old class: {} [Train size={}" \
                           "".format(
            mapped_new_classes, mapped_old_classes, int(samples_per_class), self.train_store_dict)

        if val is not None:
            val_x, val_y = val
            assert len(set(val_y)) == len(set(train_y))
            val_dict = self.get_dict_by_class(val_x, val_y)
            val_y_counts = Counter(val_y)
            self.val_store_dict = self.get_holdout_size_by_labels(val_y_counts, val_store_num, val=True)
            exemplar_details += f", Val sizes: {self.val_store_dict}"

        exemplar_details += "]\n"
        file = open(self.outfile, 'a')
        file.write(exemplar_details)
        file.close()
        print(exemplar_details)

        if strategy == 'icarl':
            self.icarl_update(train_dict, val_dict=val_dict)
        elif strategy == 'random':
            self.random_update(train_dict, val_dict=val_dict)
        elif strategy == 'sensitivity':
            if task_num == 0:
                # do a random selection for first task
                self.random_update(train_dict, val_dict=val_dict)
            elif task_num > 0:
                self.sensitivity_update(train_dict, val_dict=val_dict)
        elif strategy == 'fwsr':
            self.perform_fwsr_update(train_dict, val_dict=val_dict)
        elif strategy == 'boundary':
            if task_num == 0:
                # do a random selection for first task
                self.random_update(train_dict, val_dict=val_dict)
            else:
                self.boundary_update(train_dict, val_dict=val_dict)
        else:
            # for kmeans and dpp strategies
            self.clustered_update(strategy, train_dict, val_dict=val_dict)

        if val is not None:
            print(f' {self.cur_cls}, {len(list(self.val.keys()))}, class num: {cls_num},'
                  f' val_store_nums: {self.val_store_dict}, len: {len(list(self.val.values())[0])}')
            assert self.cur_cls == len(list(self.val.keys()))
            for key, value in self.val.items():
                assert len(self.val[key]) == self.val_store_dict[key], print(key, len(self.val[key]),
                                                                             self.val_store_dict[key])

        assert self.cur_cls % len(list(self.train.keys())) == 0 # for permuted mnist, seen cls = always multiple of total classes
        total_size = 0
        for key, value in self.train.items():
            total_size += len(value)
            print(f"Total exemplar size: {total_size}")
            assert len(self.train[key]) == self.train_store_dict[key], print(key, len(self.train[key]),
                                                                             self.train_store_dict[key])
            print(f"Class: {key}, No. of exemplars: {len(value)}")

        if vis:
            exemplar_visualizer.scatter_plot_exemps(train_dict, self.train_to_indices, self.virtual_mapping,
                                                    self.original_mapping, strategy, self.dataname)

    def get_exemplar_train(self):
        exemplar_train_x = []
        exemplar_train_y = []
        for key, value in self.train.items():
            for train_x in value:
                exemplar_train_x.append(train_x)
                exemplar_train_y.append(key)
        return exemplar_train_x, exemplar_train_y

    def get_exemplar_val(self, classes_to_exclude=[]):
        exemplar_val_x = []
        exemplar_val_y = []
        for key, value in self.val.items():
            if key not in classes_to_exclude:
                for val_x in value:
                    exemplar_val_x.append(val_x)
                    exemplar_val_y.append(key)
        return exemplar_val_x, exemplar_val_y

    def get_cur_cls(self):
        return self.cur_cls
