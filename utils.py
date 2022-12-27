#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : utils.py
# @Time      : 2022/01/01 22:15:14
# @Author    : Zhao-Wenny

import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve


def seed_torch(seed):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def plot_average_PR_curve(pred_all, label_all, fig_save_path):
    """Computes the average PR curve across the different cross validation runs.

    This function uses the results from compute_ensemble_predictions and plots
    a ROC curve from that information. The curve depicts also confidence
    intervalls and the average curve as well as AUPR values.

    Parameters:
    ----------
    model_dir:                  The output directory of the GCN training
    pred_all:                   A list with the predictions for all folds
                                (First return value of compute_ensemble_predictions)
    label_all:                   The different test sets for all folds as list
                                (Second return value of compute_ensemble_predictions)
    """
    fig = plt.figure(figsize=(20, 12))

    y_true = []
    y_pred = []
    pr_values = []
    rec_values = []

    sample_thresholds = np.linspace(0, 1, 100)
    for i in range(len(pred_all)):
        pred = pred_all[i]
        y_t = label_all[i]
        pr, rec, thr = precision_recall_curve(y_t, pred)
        pr_values.append(np.interp(sample_thresholds, thr, pr[:-1]))
        rec_values.append(np.interp(sample_thresholds, thr, rec[:-1]))
        aupr = auc(rec, pr)
        plt.plot(rec, pr, lw=4, alpha=0.4,
                 label='Fold %d (AUPR = %0.2f)' % (i, aupr))

        y_true.append(y_t)
        y_pred.append(pred)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    mean_precision, mean_recall, _ = precision_recall_curve(
        y_true, y_pred)

    mean_aupr = auc(mean_recall, mean_precision)

    label = f'Mean PR (AUPR={mean_aupr:.2f})'
    plt.plot(mean_recall, mean_precision, alpha=.8,
             label=label, linestyle='--', lw=6, color='darkred')

    # plot std dev
    std_pr = np.std(pr_values, axis=0)
    mean_pr = np.mean(pr_values, axis=0)
    mean_rec = np.mean(rec_values, axis=0)
    pr_upper = np.minimum(mean_pr + std_pr, 1)
    pr_lower = np.maximum(mean_pr - std_pr, 0)
    pr_upper = np.append(pr_upper, 1.)
    pr_lower = np.append(pr_lower, 1.)
    mean_rec = np.append(mean_rec, 0.)

    plt.fill_between(mean_rec, pr_lower, pr_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.tick_params(axis='both', labelsize=20)
    plt.xlabel('Recall', fontsize=25)
    plt.ylabel('Precision', fontsize=25)
    plt.legend(prop={'size': 20})
    fig.savefig(os.path.join(fig_save_path, 'mean_PR_curve.svg'), dpi=300)
    plt.close(fig=fig)


def plot_average_ROC_curve(pred_all, label_all, fig_save_path):

    fig = plt.figure(figsize=(20, 12))

    tprs = []
    aucs = []

    mean_fpr = np.linspace(0, 1, 100)
    for i in range(len(pred_all)):
        pred = pred_all[i]
        y_t = label_all[i]
        fpr, tpr, _ = roc_curve(y_t, pred)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[0][0] = 0.0
        auroc = roc_auc_score(y_t, pred)
        aucs.append(auroc)
        plt.plot(fpr, tpr, lw=4, alpha=0.3,
                 label='Fold %d (AUROC = %0.2f)' % (i, auroc))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[0] = 0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.plot(mean_fpr, mean_tpr, color='navy',
             label=r'Mean ROC curve(AUC = %0.2f $\pm$ %0.2f)' % (
                 mean_auc, std_auc),
             lw=6, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=25)
    plt.ylabel('True Positive Rate', fontsize=25)

    plt.legend(loc="lower right", prop={'size': 20})
    plt.tick_params(axis='both', labelsize=20)
    fig.savefig(os.path.join(fig_save_path, 'mean_ROC_curve.svg'), dpi=300)
    plt.close(fig=fig)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        self.val_loss_min = val_loss


def one_hot(labels, n_classes=2):
    return torch.eye(n_classes)[labels, :]


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
