#!/usr/bin/env python3

"""Functions for computing metrics. all metrics has range of 0-1"""
import logging
import os.path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, roc_auc_score,roc_curve
)


def accuracy(y_probs, y_true):
    # y_prob: (num_images, num_classes)
    y_preds = np.argmax(y_probs, axis=1)
    accuracy = accuracy_score(y_true, y_preds)
    error = 1.0 - accuracy
    return accuracy, error


def top_n_accuracy(y_probs, truths, n=1):
    # y_prob: (num_images, num_classes)
    # truth: (num_images, num_classes) multi/one-hot encoding
    best_n = np.argsort(y_probs, axis=1)[:, -n:]
    if isinstance(truths, np.ndarray) and truths.shape == y_probs.shape:
        ts = np.argmax(truths, axis=1)
    else:
        # a list of GT class idx
        ts = truths

    num_input = y_probs.shape[0]
    successes = 0
    for i in range(num_input):
        if ts[i] in best_n[i, :]:
            successes += 1
    return float(successes) / num_input


def compute_acc_auc(y_probs, y_true_ids,nameList=None):
    onehot_tgts = np.zeros_like(y_probs)
    # print(y_true_ids)
    #
    # print(y_probs)
    for idx, t in enumerate(y_true_ids):
        onehot_tgts[idx, t] = 1.

    num_classes = y_probs.shape[1]
    if num_classes == 2:
        top1, _ = accuracy(y_probs, y_true_ids)
        # so precision can set all to 2
        try:
            auc = roc_auc_score(onehot_tgts, y_probs, average='macro')
        except ValueError as e:
            print(f"value error encountered {e}, set auc sccore to -1.")
            auc = -1
        # probability = torch.softmax(y_probs, dim=1)[:, 1].detach().tolist()
        # try:
        y_probs_softmax = torch.softmax(torch.tensor(y_probs), 1)[:, 1].detach().tolist()
        y_true_ids = np.array(y_true_ids)
        # print(len(y_true_ids), y_probs[:,1].shape,y_probs_softmax)
        fpr, tpr, thresholds = roc_curve(y_true_ids, y_probs_softmax)
        eer_threshold = thresholds[np.nanargmin(np.absolute((1 - tpr) - fpr))]
        eer = (np.sum(y_true_ids[np.where(y_probs_softmax < eer_threshold)] == 1) / np.sum(y_true_ids == 1) +
               np.sum(y_true_ids[np.where(y_probs_softmax >= eer_threshold)] == 0) / np.sum(y_true_ids == 0)) / 2
        print(fpr,tpr)
        print(eer,":eer","eer_threshold",eer_threshold)
        # print("nameList",nameList)
        # except ValueError as e:
        #     print(f"value error encountered {e}, set eer sccore to -1.")
        #     eer = -1
        if nameList is not None:
            batch_results = []
            batch_eeror_results = []
            for name_, probability_, label_ in zip(nameList, y_probs_softmax, y_true_ids):
                if probability_ > eer_threshold:
                    batch_results.append((name_, probability_, label_.item(),1))
                    if label_.item() == 0:
                        batch_eeror_results.append((name_, probability_, label_.item(),1))
                else:
                    batch_results.append((name_, probability_, label_.item(), 0))
                    if label_.item() == 1:
                        batch_eeror_results.append((name_, probability_, label_.item(),0))
            def write_judge_csv(results, file_name):
                import csv
                with open(file_name, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['name', 'score', 'label', "Result"])
                    writer.writerows(results)

            resultPath = r"E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\5_29_test\doc2train_prompt_deep_test_doc1_Dir_deep_100"
            csvResultPath = os.path.join(resultPath,"batchResults.csv")
            csvEERResultPath = os.path.join(resultPath, "batchEERResults.csv")
            write_judge_csv(batch_results,csvResultPath)
            write_judge_csv(batch_eeror_results, csvEERResultPath)
            print('csvResultPath',csvResultPath,'csvEERResultPath',csvEERResultPath)
        return {"top1": top1, "rocauc": auc,"eer":eer}

    top1, _ = accuracy(y_probs, y_true_ids)
    k = min([5, num_classes])  # if number of labels < 5, use the total class
    top5 = top_n_accuracy(y_probs, y_true_ids, k)
    return {"top1": top1, f"top{k}": top5}


def topks_correct(preds, labels, ks):
    """Computes the number of top-k correct predictions for each k."""
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size)
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size)
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k
    topks_correct = [
        top_max_k_correct[:k, :].reshape(-1).float().sum() for k in ks
    ]
    return topks_correct


def topk_errors(preds, labels, ks):
    """Computes the top-k error for each k."""
    if int(labels.min()) < 0:  # has ignore
        keep_ids = np.where(labels.cpu() >= 0)[0]
        preds = preds[keep_ids, :]
        labels = labels[keep_ids]

    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """Computes the top-k accuracy for each k."""
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) for x in num_topks_correct]

