#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : main.py
# @Time      : 2022/01/01 22:20:17
# @Author    : Zhao-Wenny

import argparse
import os

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.tensorboard.writer import SummaryWriter

from modig import MODIG
from modig_graph import ModigGraph
from utils import *

cuda = torch.cuda.is_available()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train MODIG with cross-validation and save model to file')
    parser.add_argument('-t', '--title', help='the name of running experiment',
                        dest='title',
                        default=None,
                        type=str
                        )
    parser.add_argument('-ppi', '--ppi', help='the chosen type of PPI',
                        dest='ppi',
                        default='CPDB',
                        type=str
                        )
    parser.add_argument('-omic', '--omic', help='the chosen node attribute [multiomic, snv, cnv, mrna, dm]',
                        dest='omic',
                        default='multiomic',
                        type=str
                        )
    parser.add_argument('-cancer', '--cancer', help='the model on pancan or specific cancer type',
                        dest='cancer',
                        default='pancan',
                        type=str
                        )
    parser.add_argument('-e', '--epochs', help='maximum number of epochs (default: 1000)',
                        dest='epochs',
                        default=1000,
                        type=int
                        )
    parser.add_argument('-p', '--patience', help='patience (default: 20)',
                        dest='patience',
                        default=20,
                        type=int
                        )
    parser.add_argument('-dp', '--dropout', help='the dropout rate (default: 0.25)',
                        dest='dp',
                        default=0.25,
                        type=float
                        )
    parser.add_argument('-lr', '--learningrate', help='the learning rate (default: 0.001)',
                        dest='lr',
                        default=0.001,
                        type=float
                        )
    parser.add_argument('-wd', '--weightdecay', help='the weight decay (default: 0.0005)',
                        dest='wd',
                        default=0.0005,
                        type=float
                        )
    parser.add_argument('-hs1', '--hiddensize1', help='the hidden size of first convolution layer (default: 300)',
                        dest='hs1',
                        default=300,
                        type=int
                        )
    parser.add_argument('-hs2', '--hiddensize2', help='the hidden size of second convolution layer (default: 100)',
                        dest='hs2',
                        default=100,
                        type=int
                        )
    parser.add_argument('-thr_go', '--thr_go', help='the threshold for GO semantic similarity (default: 0.8)',
                        dest='thr_go',
                        default=0.8,
                        type=float
                        )
    parser.add_argument('-thr_seq', '--thr_seq', help='the threshold for gene sequence similarity (default: 0.5)',
                        dest='thr_seq',
                        default=0.5,
                        type=float
                        )
    parser.add_argument('-thr_exp', '--thr_exp', help='the threshold for tissue co-expression pattern (default: 0.8)',
                        dest='thr_exp',
                        default=0.8,
                        type=float
                        )
    parser.add_argument('-thr_path', '--thr_path', help='the threshold of gene pathway co-occurrence (default: 0.5)',
                        dest='thr_path',
                        default=0.5,
                        type=float
                        )
    parser.add_argument('-seed', '--seed', help='the random seed (default: 42)',
                        dest='seed',
                        default=42,
                        type=int
                        )
    args = parser.parse_args()
    return args


def main(args):

    seed_torch(args['seed'])
    file_save_path = os.path.join('./Output', args['ppi'], args['title'])
    make_dir(file_save_path)

    # load data
    graph_path = os.path.join('./Data/graph', args['ppi'])
    if not os.path.exists(graph_path):
        os.makedirs(graph_path)

    modig_input = ModigGraph(graph_path, args['ppi'], args['cancer'])

    print('Network INFO')

    ppi_path = os.path.join(graph_path, args['ppi'] + '_ppi.tsv')
    go_path = os.path.join(
        graph_path, args['ppi'] + '_' + str(args['thr_go']) + '_go.tsv')
    exp_path = os.path.join(
        graph_path, args['ppi'] + '_' + str(args['thr_exp']) + '_exp.tsv')
    seq_path = os.path.join(
        graph_path, args['ppi'] + '_' + str(args['thr_seq']) + '_seq.tsv')
    path_path = os.path.join(
        graph_path, args['ppi'] + '_' + str(args['thr_path']) + '_path.tsv')

    omic_path = os.path.join(graph_path, args['ppi'] + '_omics.tsv')

    if os.path.exists(ppi_path) & os.path.exists(go_path) & os.path.exists(exp_path) & os.path.exists(seq_path) & os.path.exists(path_path) & os.path.exists(omic_path):
        print('The five gene similarity profiles and omic feature already exist!')
        ppi_network = pd.read_csv(ppi_path, sep='\t', index_col=0)
        go_network = pd.read_csv(go_path, sep='\t', index_col=0)
        exp_network = pd.read_csv(exp_path, sep='\t', index_col=0)
        seq_network = pd.read_csv(seq_path, sep='\t', index_col=0)
        path_network = pd.read_csv(path_path, sep='\t', index_col=0)
        omicsfeature = pd.read_csv(omic_path, sep='\t', index_col=0)
        final_gene_node = list(omicsfeature.index)

    else:
        omicsfeature, final_gene_node = modig_input.get_node_omicfeature()
        ppi_network, go_network, exp_network, seq_network, path_network = modig_input.generate_graph(
            args['thr_go'], args['thr_exp'], args['thr_seq'], args['thr_path'])

    print("==========================================================")
    print('Network INFO')
    name_of_network = ['PPI', 'GO', 'EXP', 'SEQ', 'PATH']
    graphlist = []
    for i, network in enumerate([ppi_network, go_network, exp_network, seq_network, path_network]):
        featured_graph = modig_input.load_featured_graph(network, omicsfeature)
        print(f'The {name_of_network[i]} graph: {featured_graph}')
        graphlist.append(featured_graph)

    n_fdim = graphlist[0].x.shape[1]  # n_gene = featured_gsn.x.shape[0]
    graphlist_adj = [graph.cuda() for graph in graphlist]
    k_sets, idx_list, label_list = modig_input.ten_fold_five_crs_validation(
        file_save_path)
    print("==========================================================")

    def train(mask, label):
        model.train()
        optimizer.zero_grad()
        output = model(graphlist_adj)
        loss = F.binary_cross_entropy_with_logits(
            output[mask], label, pos_weight=torch.Tensor([2.7]).cuda())

        acc = metrics.accuracy_score(label.cpu(), np.round(
            torch.sigmoid(output[mask]).cpu().detach().numpy()))
        loss.backward()
        optimizer.step()

        del output
        return loss.item(), acc

    @torch.no_grad()
    def test(mask, label):
        model.eval()
        output = model(graphlist_adj)
        loss = F.binary_cross_entropy_with_logits(
            output[mask], label, pos_weight=torch.Tensor([2.7]).cuda())

        acc = metrics.accuracy_score(label.cpu(), np.round(
            torch.sigmoid(output[mask]).cpu().detach().numpy()))
        pred = torch.sigmoid(output[mask]).cpu().detach().numpy()
        auroc = metrics.roc_auc_score(label.to('cpu'), pred)
        pr, rec, _ = metrics.precision_recall_curve(label.to('cpu'), pred)
        aupr = metrics.auc(rec, pr)

        return pred, loss.item(), acc, auroc, aupr

    AUC = np.zeros(shape=(10, 5))
    AUPR = np.zeros(shape=(10, 5))
    ACC = np.zeros(shape=(10, 5))

    pred_all = []
    label_all = []

    for j in range(len(k_sets)):
        print(j)
        for cv_run in range(5):
            train_mask, val_mask, train_label, val_label = [
                p.cuda() for p in k_sets[j][cv_run] if type(p) == torch.Tensor]

            model = MODIG(
                nfeat=n_fdim, hidden_size1=args['hs1'], hidden_size2=args['hs2'], dropout=args['dp'])
            model.cuda()
            optimizer = optim.Adam(
                model.parameters(), lr=args['lr'], weight_decay=args['wd'])
            # model_save_file = os.path.join(log_dir, str(cv_run) + '_modig.pth')

            early_stopping = EarlyStopping(
                patience=args['patience'], verbose=True)

            for epoch in range(1, args['epochs']+1):
                _, _ = train(train_mask, train_label)
                _, loss_val, _, _, _ = test(val_mask, val_label)

                early_stopping(loss_val, model)
                if early_stopping.early_stop:
                    print(f"Early stopping at the epoch {epoch}")
                    break

                torch.cuda.empty_cache()

            pred, _, ACC[j][cv_run], AUC[j][cv_run], AUPR[j][cv_run] = test(
                val_mask, val_label)

            pred_all.append(pred)
            label_all.append(val_label.to('cpu'))

    print('Mean AUC', AUC.mean())
    print('Var AUC', AUC.var())
    print('Mean AUPR', AUPR.mean())
    print('Var AUPR', AUPR.var())
    print('Mean ACC', ACC.mean())
    print('Var ACC', ACC.var())

    torch.save(pred_all, os.path.join(file_save_path, 'pred_all.pkl'))
    torch.save(label_all, os.path.join(file_save_path, 'label_all.pkl'))

    # Use all label to train a final model
    all_mask = torch.LongTensor(idx_list)
    all_label = torch.FloatTensor(label_list).reshape(-1, 1)

    model = MODIG(nfeat=n_fdim, hidden_size1=args['hs1'],
                  hidden_size2=args['hs2'], dropout=args['dp'])
    model.cuda()
    optimizer = optim.Adam(
        model.parameters(), lr=args['lr'], weight_decay=args['wd'])

    for epoch in range(1, args['epochs']+1):
        print(epoch)
        _, _ = train(all_mask.cuda(), all_label.cuda())
        torch.cuda.empty_cache()

    output = model(graphlist_adj)

    pred = torch.sigmoid(output).cpu().detach().numpy()
    pred2 = torch.sigmoid(output[~all_mask]).cpu().detach().numpy()
    torch.save(pred, os.path.join(file_save_path, args['ppi'] + '_pred.pkl'))
    torch.save(all_label, os.path.join(
        file_save_path, args['ppi'] + '_label.pkl'))
    torch.save(pred2, os.path.join(file_save_path, args['ppi'] + '_pred2.pkl'))

    pd.Series(final_gene_node).to_csv(os.path.join(file_save_path,
                                                   'final_gene_node.csv'), index=False, header=False)

    plot_average_PR_curve(pred_all, label_all, file_save_path)
    plot_average_ROC_curve(pred_all, label_all, file_save_path)


if __name__ == '__main__':

    args = parse_args()
    args_dic = vars(args)
    print('args_dict', args_dic)

    main(args_dic)
    print('The Training is finished!')
