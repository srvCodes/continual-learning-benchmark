import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cosine
from torch.autograd import Variable
from tqdm import tqdm

from train.losses import class_balanced_loss
from .ewc import EWC
from .losses import margin_ranking_loss

cur_features = []
ref_features = []
old_scores = []
new_scores = []
lpl_features_student, lpl_features_teacher = [], []


def get_ref_features(self, inputs, outputs):
    global ref_features
    ref_features = inputs[0]


def get_cur_features(self, inputs, outputs):
    global cur_features
    cur_features = inputs[0]


def get_old_scores_before_scale(self, inputs, outputs):
    global old_scores
    old_scores = outputs


def get_new_scores_before_scale(self, inputs, outputs):
    global new_scores
    new_scores = outputs


def get_teacher_features(self, inputs, outputs):
    global lpl_features_teacher
    lpl_features_teacher = outputs


def get_student_features(self, inputs, outputs):
    global lpl_features_student
    lpl_features_student = outputs


loss_by_epoch = dict()


class CustomizedTrainer():
    def __init__(self, args, itera, seen_cls, train_loader, model, previous_model, lamda,
                 bias_layers, virtual_map, classes_by_groups, device, visualizer=None, old_tasks=None):
        self.args = args
        self.itera = itera
        self.seen_class = seen_cls
        self.train_loader = train_loader
        self.device = device
        self.cur_lamda = lamda
        self.model, self.previous_model = model, previous_model
        self.data_visualizer = visualizer
        self.virtual_map = virtual_map
        self.handle_ref_features = self.previous_model.fc.register_forward_hook(get_ref_features)
        self.handle_cur_features = self.model.fc.register_forward_hook(get_cur_features)
        if 'cn' in self.args.method:
            self.handle_old_scores_bs = self.model.fc.fc1.register_forward_hook(get_old_scores_before_scale)
            self.handle_new_scores_bs = self.model.fc.fc2.register_forward_hook(get_new_scores_before_scale)
        if 'bic' in self.args.method:
            self.bias_layers = bias_layers
            self.classes_by_groups = classes_by_groups
        if 'ewc' in self.args.method:
            self.old_tasks = old_tasks
        if 'lpl' in self.args.method:
            self.handle_student_features = self.model.fc_penultimate.register_forward_hook(get_student_features)
            self.handle_teacher_features = self.model.fc_penultimate.register_forward_hook(get_teacher_features)

    def align_weights(self):
        print("Aligning weights: ")
        if 'wa1' in self.args.method:
            with torch.no_grad():
                for p in self.model.fc.parameters():
                    p.data.clamp_(0)
        elif 'wa2' in self.args.method:
            if 'cn' in self.args.method:
                # customize the normalization so as to fit for the fc1 and fc2 layers of SplitCosineLayer()
                with torch.no_grad():
                    w1, w2 = self.model.fc.fc1.weight.data, self.model.fc.fc2.weight.data
                    w1_norm, w2_norm = [w / torch.norm(w, p=2, dim=1, keepdim=True) for w in (w1, w2)]
                    self.model.fc.fc1.weight.data.copy_(w1_norm)
                    self.model.fc.fc2.weight.data.copy_(w2_norm)
            else:
                # carry out regular normalization with fc layers
                with torch.no_grad():
                    # https://github.com/feidfoe/AdjustBnd4Imbalance/blob/master/models/cifar/resnet.py#L135
                    w = self.model.fc.weight.data
                    w_norm = w / torch.norm(w, p=2, dim=1, keepdim=True)
                    self.model.fc.weight.data.copy_(w_norm)

    def distill_training(self, optimizer, num_new_classes, last_epoch=False, new_class_avg=None):
        print("Training with distillation losses... ")
        losses = []
        lambda_ = (self.seen_class - num_new_classes) / self.seen_class
        dataloader = self.train_loader
        if self.args.tsne_vis and last_epoch:
            tsne_features, tsne_labels = np.empty(shape=[0, self.model.fc.in_features]), []

        if 'ewc' in self.args.method:
            ewc = EWC(self.model, self.old_tasks, self.device)

        for i, (feature, label) in enumerate(tqdm(dataloader)):
            feature, label = Variable(feature), Variable(label)
            feature = feature.to(self.device)
            # label = label.type(torch.LongTensor)
            label = label.view(-1).to(self.device)
            optimizer.zero_grad()
            p = self.model(feature.float())
            if 'bic' in self.args.method and self.itera > 1:
                p = self.bias_forward(p)

            logp = F.log_softmax(p[:, :self.seen_class - num_new_classes] / self.args.T, dim=1)

            with torch.no_grad():
                if self.args.weighted:
                    sample_weights = self.get_sample_weights(feature, label, new_class_avg, num_new_classes)
                    sample_weights = torch.exp(torch.Tensor(sample_weights).to(self.device))
                pre_p = self.previous_model(feature.float())
                if 'bic' in self.args.method and self.itera > 1:
                    pre_p = self.bias_forward(pre_p)
                # apply distilling loss giving soft labels (T = wts. of small values to introduce rescaling)
                pre_p = F.softmax(pre_p[:, :self.seen_class - num_new_classes] / self.args.T, dim=1)

            if self.args.extra_loss == 'modified_cd':
                p = self.modify_new_logits(p, pre_p, num_new_classes)

            loss_hard_target = torch.nn.CrossEntropyLoss()(p[:, :self.seen_class],
                                                           label)
            if any([x in self.args.method for x in ['ce', 'cn']]):
                # for both base_ce and while using cosine normalisation, whole of the loss_hard_target is added
                loss = loss_hard_target
                loss_stats = f"CE loss: {loss_hard_target}"
            else:
                print("No valid distill method: 'ce' or 'kd' or 'cn' found !!!")
            if 'lfc' in self.args.method:
                # less forget constraint loss
                cur_features_ = F.normalize(cur_features, p=2, dim=1)
                ref_features_ = F.normalize(ref_features.detach(), p=2, dim=1)
                less_forget_constraint = torch.nn.CosineEmbeddingLoss()(cur_features_, ref_features_,
                                                                        torch.ones(feature.shape[0]).to(
                                                                            self.device)) * self.cur_lamda
                loss += less_forget_constraint
                loss_stats += f" LFC loss: {less_forget_constraint}"
            if 'mr' in self.args.method:
                # compute margin ranking loss
                if 'cn' in self.args.method:
                    output_bs = torch.cat((old_scores, new_scores), dim=1)
                else:
                    output_bs = p
                    output_bs = F.normalize(output_bs, p=2, dim=1)
                mr_loss = margin_ranking_loss.compute_margin_ranking_loss(p, label, num_new_classes, self.seen_class,
                                                                          self.device, output_bs)
                loss += mr_loss
                loss_stats += f" MR loss: {mr_loss}"
            if 'lpl' in self.args.method:
                # locality preserving loss
                k, gamma = 5 if len(label) > 5 else math.ceil(len(label) / 2), 1.5
                lpl_loss = 0
                for i, data in enumerate(lpl_features_student):
                    f_s_i = data
                    for j, data_ in enumerate(lpl_features_student):
                        if j != i:
                            alpha_i_j= self.get_locality_preserving_alpha(i, j, k)
                            if alpha_i_j > 0:
                                temp_ = torch.norm(f_s_i - data_, dim=0, p=None).pow(2)
                                lpl_loss += temp_.item() * alpha_i_j
                lpl_loss = gamma * lpl_loss / (label.shape[0] * k)  # scale by factor: gamma / (k * batch_size)
                loss += lpl_loss
                loss_stats += f" LPL loss: {lpl_loss}"

            print(loss_stats)
            if 'wa' in self.args.method:
                self.align_weights()

            loss.backward(retain_graph=True)
            optimizer.step()
            # look into tsne visualisations
            if self.args.tsne_vis and last_epoch:
                tsne_features = np.vstack((tsne_features, cur_features.detach().cpu().numpy()))
                tsne_labels += label.cpu().numpy().tolist()
                self.data_visualizer.plot_tsne(tsne_features, tsne_labels, itera=self.itera)
            losses.append(loss.item())

        return sum(losses) / len(dataloader.dataset)

    def get_locality_preserving_alpha(self, i, j, k=5):
        sigma = math.sqrt(2) # normalizing constant
        f_T_i = lpl_features_teacher[i]
        dist = torch.norm(lpl_features_teacher - f_T_i, dim=1, p=None)
        knn_indices = dist.topk(k+1, largest=False).indices[1:] # 0th index is always the element itself
        if j in knn_indices:
            alpha_i_j = - dist[j].float().pow(2) / sigma ** 2
            alpha_i_j = torch.exp(alpha_i_j).item()
        else:
            alpha_i_j = 0.
        return alpha_i_j

    def modify_new_logits(self, p, p_old, m):
        """
        Adapted from https://arxiv.org/pdf/2003.13191.pdf
        :param p: output logits of new classifier (o_1...o_n, o_n+1...o_n+m)
        :param p_old: old classifier output logits (o_1...o_n)
        :param m: num of new classes
        :return: modified logits of new classifier
        """
        beta = 0.8  # beta = 0.5 used in the paper
        p[:, :self.seen_class - m] = p[:, :self.seen_class - m] * beta + p_old * (1 - beta)
        return p

    def bias_forward(self, input):
        input_groups = []
        for idx, classes in enumerate(self.classes_by_groups):
            temp_tensor = torch.Tensor().to(self.device)
            for each in classes:
                each = self.virtual_map[each]
                temp_tensor = torch.cat([temp_tensor, input[:, (int)(each):(int)(each + 1)]], dim=1)
            input_groups.append(temp_tensor)
        output_by_groups = [self.bias_layers[idx](item) for idx, item in enumerate(input_groups)]
        output_by_groups = torch.cat(output_by_groups, dim=1)
        return output_by_groups

    @staticmethod
    def get_count_by_classes(array_of_labels, seen_classes):
        classes_seen = [i for i in range(seen_classes)]
        counts = []
        for label in classes_seen:
            counts.append(np.count_nonzero(array_of_labels == label))
        assert len(counts) == seen_classes
        return counts

    def get_sample_weights(self, features, labels, label_averaged_dict, num_new_classes):
        # feature_extractor = basic_model.Net(self.model.fc1.in_features, self.args.total_classes)
        # feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])
        past_classes = [i for i in range(self.seen_class - num_new_classes)]
        max_elem = 0
        batch_distance = []
        for feature, label in zip(cur_features, labels):
            feature = feature.div(feature.norm(p=2, dim=0, keepdim=True).expand_as(feature))
            if label in past_classes:
                cosine_dist = tuple((cosine(feature.detach().cpu(), averaged_vec) for _, averaged_vec in
                                     label_averaged_dict.items()))
                batch_distance.append(np.array([max(cosine_dist)]))
            else:
                batch_distance.append(np.array([0]))
        max_elem = max(batch_distance)
        assert (len(batch_distance) == len(features))
        distances = list(map(lambda x: x - max_elem, batch_distance))  # x = x - max(x_i) for avoiding underflow
        return distances

    def remove_hooks(self):
        # remove the registered hook after model has been trained for the incremental batch
        self.handle_ref_features.remove()
        self.handle_cur_features.remove()
        if 'cn' in self.args.method:
            self.handle_old_scores_bs.remove()
            self.handle_new_scores_bs.remove()
        if 'lpl' in self.args.method:
            self.handle_student_features.remove()
            self.handle_teacher_features.remove()

    def get_model(self):
        return self.model