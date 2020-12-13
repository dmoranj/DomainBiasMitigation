import os
import json
import pickle
import numpy as np
import torch
from sklearn.metrics import average_precision_score

def save_pkl(pkl_data, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(pkl_data, f)

def load_pkl(load_path):
    with open(load_path, 'rb') as f:
        pkl_data = pickle.load(f)
    return pkl_data

def save_json(json_data, save_path):
    with open(save_path, 'w') as f:
        json.dump(json_data, f)

def load_json(load_path):
    with open(load_path, 'r') as f:
        json_data = json.load(f)
    return json_data

def save_state_dict(state_dict, save_path):
    torch.save(state_dict, save_path)

def creat_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def set_random_seed(seed_number):
    torch.manual_seed(seed_number)
    np.random.seed(seed_number)
    
def write_info(filename, info):
    with open(filename, 'w') as f:
        f.write(info)


def compute_weighted_AP(target, predict_prob, class_weight_list):
    per_class_AP = []
    for i in range(target.shape[1] - 1):
        class_weight = target[:, i]*class_weight_list[i] \
                       + (1-target[:, i])*np.ones(class_weight_list[i].shape)
        per_class_AP.append(average_precision_score(target[:, i], predict_prob[:, i], 
                                sample_weight=class_weight))
        
    return per_class_AP


def compute_mAP(per_class_AP, subclass_idx):
    return np.mean([per_class_AP[idx] for idx in subclass_idx])


def compute_matrix_metrics(predicted, targets, classes=10):
    tp_per_class = np.array([(predicted[targets.eq(class_id)] == class_id).sum().item() for class_id in range(classes)])
    fn_per_class = np.array([(predicted[targets.eq(class_id)] != class_id).sum().item() for class_id in range(classes)])
    fp_per_class = np.array(
        [(predicted[targets.eq(class_id).logical_not()] == class_id).sum().item() for class_id in range(classes)])

    precision_per_class, recall_per_class = compute_precision_and_recall(fn_per_class, fp_per_class, tp_per_class)

    return precision_per_class, recall_per_class


def compute_precision_and_recall(fn_per_class, fp_per_class, tp_per_class):
    total_positives = tp_per_class + fn_per_class
    total_predicted = tp_per_class + fp_per_class
    tpos_not_zero = total_positives > 0
    tpred_not_zero = total_predicted > 0
    precision_per_class = tp_per_class[tpred_not_zero] / total_predicted[tpred_not_zero]
    recall_per_class = tp_per_class[tpos_not_zero] / total_positives[tpos_not_zero]
    return precision_per_class, recall_per_class


def compute_matrix_metrics_multilabel(predicted, targets):
    tp = (predicted + targets == 2).sum(1)
    fp = (targets - predicted == -1).sum(1)
    fn = (predicted - targets == -1).sum(1)
    precision, recall = compute_precision_and_recall(fn, fp, tp)

    return precision, recall


def compute_metrics(target, predicted_prob, threshold=0.5):
    target_tensor = torch.tensor(target[:, :-1])
    predictions_tensor = (torch.tensor(predicted_prob) > threshold).int()

    precision, recall = compute_matrix_metrics_multilabel(predictions_tensor, target_tensor)
    precision = np.nanmean(precision)
    recall = np.nanmean(recall)

    f1 = 0.5 * (precision * recall) / (precision + recall)

    return f1, precision, recall


def compute_class_weight(target):
    domain_label = target[:, -1]
    per_class_weight = []

    for i in range(target.shape[1]-1):
        class_label = target[:, i]
        cp = class_label.sum() # class is positive
        cn = target.shape[0] - cp # class is negative
        cn_dn = ((class_label + domain_label)==0).sum() # class is negative, domain is negative
        cn_dp = ((class_label - domain_label)==-1).sum()
        cp_dn = ((class_label - domain_label)==1).sum()
        cp_dp = ((class_label + domain_label)==2).sum()

        per_class_weight.append(
            (class_label*cp + (1-class_label)*cn) / 
                (2*(
                    (1-class_label)*(1-domain_label)*cn_dn
                    + (1-class_label)*domain_label*cn_dp
                    + class_label*(1-domain_label)*cp_dn
                    + class_label*domain_label*cp_dp
                   )
                )
        )
    return per_class_weight