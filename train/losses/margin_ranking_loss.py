import torch
import math

def compute_margin_ranking_loss(logits, minibatch_labels, num_new_classes, seen_classes, device, outputs_bs):
    K, dist, lw_mr = 2, 0.5, 1
    # get scores before [-1,1] scaling
    assert (outputs_bs.size() == logits.size())
    # compute ground truth scores
    high_response_index = torch.zeros(outputs_bs.size()).to(device)
    high_response_index = high_response_index.scatter(1, minibatch_labels.view(-1, 1), 1).ge(dist) # indices of actual class = True, rest = False
    high_response_scores = outputs_bs.masked_select(high_response_index) # scores of actual labels
    # compute top-K scores on none high response classes
    none_gt_index = torch.zeros(outputs_bs.size()).to(device)
    none_gt_index = none_gt_index.scatter(1, minibatch_labels.view(-1, 1), 1).le(dist) # True for all negative classes, False for actual class
    none_gt_scores = outputs_bs.masked_select(none_gt_index).reshape((outputs_bs.size(0), logits.size(1) - 1)) # scores for negative classes
    hard_negatives_scores = none_gt_scores.topk(K, dim=1)[0] # top k negative classes
    hard_negatives_index = minibatch_labels.lt(seen_classes - num_new_classes) # True for all old classes
    hard_negatives_num = torch.nonzero(hard_negatives_index).size(0) # number of old class labels
    if hard_negatives_num > 0:
        gt_scores = high_response_scores[hard_negatives_index].view(-1, 1).repeat(1, K) # logits score for old class instances repeated k times
        hard_scores = hard_negatives_scores[hard_negatives_index] # score for top-k instances the old class is most confused with
        assert (gt_scores.size() == hard_scores.size())
        assert (gt_scores.size(0) == hard_negatives_num)
        mr_loss = torch.nn.MarginRankingLoss(margin=dist)(gt_scores.view(-1, 1), hard_scores.view(-1, 1),
                                                          torch.ones(hard_negatives_num * K).to(device)) * lw_mr
    else:
        mr_loss = torch.tensor(0.).to(device)
    return mr_loss

def compute_triplet_loss(logits, minibatch_labels, num_new_classes, seen_classes, device, outputs_bs):
    K, dist, lw_mr = 2, 0.5, 1
    # get scores before [-1,1] scaling
    assert (outputs_bs.size() == logits.size())
    # compute ground truth scores
    high_response_index = torch.zeros(outputs_bs.size()).to(device)
    high_response_index = high_response_index.scatter(1, minibatch_labels.view(-1, 1), 1).ge(dist) # indices of actual labels
    print(high_response_index)
    high_response_scores = outputs_bs.masked_select(high_response_index)  # scores of actual labels
    # compute top-K scores on none high response classes
    none_gt_index = torch.zeros(outputs_bs.size()).to(device)
    none_gt_index = none_gt_index.scatter(1, minibatch_labels.view(-1, 1), 1).le(0.8)  # True for all negative classes
    none_gt_scores = outputs_bs.masked_select(none_gt_index).reshape(
        (outputs_bs.size(0), logits.size(1) - 1))  # scores for negative classes
    print(none_gt_index, minibatch_labels, none_gt_scores); exit(1)
    hard_negatives_scores = none_gt_scores.topk(K, dim=1)[0]  # top k negative classes
    hard_negatives_index = minibatch_labels.lt(seen_classes - num_new_classes)  # True for all old classes
    hard_negatives_num = torch.nonzero(hard_negatives_index).size(0)  # number of old class labels