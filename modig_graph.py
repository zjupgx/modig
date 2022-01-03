#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : modig_graph.py
# @Time      : 2022/01/01 22:25:04
# @Author    : Zhao-Wenny

import os

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import from_networkx


class ModigGraph(object):

    def __init__(self, graph_path, ppi_type, cancer_type):

        self.graph_path = graph_path
        self.ppi_type = ppi_type
        self.cancer_type = cancer_type

    def ten_fold_five_crs_validation(self, file_save_path, K=10, folds=5):
        # load k set label.
        final_gene_node, _ = self.get_node_genelist()

        label_file = pd.read_csv(os.path.join(
            './Data/label/' + self.cancer_type + '_genelist_for_train.tsv'), sep='\t', names=['Hugosymbol', 'Label'], header=0)
        genes_match = pd.merge(pd.Series(sorted(
            final_gene_node), name='Hugosymbol'), label_file, on='Hugosymbol', how='left')

        idx_list = np.array(genes_match[~genes_match['Label'].isnull()].index)
        print(f'The match number of gene with annotation: {len(idx_list)}')
        label_list = np.array(genes_match['Label'].loc[idx_list])
        unique, counts = np.unique(label_list, return_counts=True)
        print('The label distribution:', dict(zip(unique, counts)))

        k_sets = {}
        for i in range(K):
            kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
            splits = kf.split(idx_list, label_list)

            k_folds = []
            for train, val in splits:
                train_mask = torch.LongTensor(idx_list[train])
                val_mask = torch.LongTensor(idx_list[val])
                train_label = torch.FloatTensor(
                    label_list[train]).reshape(-1, 1)
                val_label = torch.FloatTensor(label_list[val]).reshape(-1, 1)
                k_folds.append((train_mask, val_mask, train_label, val_label))

            k_sets[i] = k_folds

        torch.save(k_sets, os.path.join(file_save_path, 'k_sets.pkl'))

        return k_sets, idx_list, label_list

    def get_node_genelist(self):
        print('Get gene list')
        gene = pd.read_csv("./Data/simmatrix/gene_info_for_GOSemSim.csv")
        gene_list = list(set(gene['Symbol']))

        ppi = pd.read_csv(os.path.join('./Data/ppi', self.ppi_type + '.tsv'), sep='\t',
                          compression='gzip', encoding='utf8', usecols=['partner1', 'partner2'])
        ppi.columns = ['source', 'target']
        ppi = ppi[ppi['source'] != ppi['target']]
        ppi.dropna(inplace=True)

        final_gene_node = sorted(
            list(set(gene_list) | set(ppi.source) | set(ppi.target)))

        return final_gene_node, ppi

    def get_node_omicfeature(self):

        final_gene_node, _ = self.get_node_genelist()

        # process the omic data
        omics_file = pd.read_csv(
            './Data/feature/biological_features.csv', sep='\t', index_col=0)

        expendgene = sorted(list(set(omics_file.index) | set(final_gene_node)))
        temp = pd.DataFrame(index=expendgene, columns=omics_file.columns)
        omics_adj = temp.combine_first(omics_file)
        omics_adj.fillna(0, inplace=True)
        omics_adj = omics_adj.loc[final_gene_node]
        omics_adj.sort_index(inplace=True)

        if self.cancer_type != 'pancan':
            omics_data = omics_adj[omics_adj.columns[omics_adj.columns.str.contains(
                self.cancer_type)]]
        elif self.cancer_type == 'pancan':
            # chosen 16 cancer type
            chosen_project = ['KIRC', 'BRCA', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD',
                              'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']
            omics_temp = [omics_adj[omics_adj.columns[omics_adj.columns.str.contains(
                cancer)]] for cancer in chosen_project]
            omics_data = pd.concat(omics_temp, axis=1)

        omics_feature_vector = sp.csr_matrix(omics_data, dtype=np.float32)
        omics_feature_vector = torch.FloatTensor(
            np.array(omics_feature_vector.todense()))
        print(
            f'The shape of omics_feature_vector:{omics_feature_vector.shape}')

        return omics_feature_vector, final_gene_node

    def generate_graph(self, thr_go, thr_exp, thr_seq, thr_path):
        """
        generate tri-graph: PPI GSN GO_network
        """
        print('generate graph')
        final_gene_node, ppi = self.get_node_genelist()

        path = pd.read_csv(os.path.join(
            './Data/simmatrix/pathsim_matrix.csv'), sep='\t', index_col=0)
        path_matrix = path.applymap(lambda x: 0 if x < thr_path else 1)
        np.fill_diagonal(path_matrix.values, 0)

        go = pd.read_csv('./Data/simmatrix/GOSemSim_matrix.csv',
                         sep='\t', index_col=0)
        go_matrix = go.applymap(lambda x: 0 if x < thr_go else 1)
        np.fill_diagonal(go_matrix.values, 0)

        exp = pd.read_csv(os.path.join(
            './Data/simmatrix/expsim_matrix.csv'), sep='\t', index_col=0)
        exp_matrix = exp.applymap(lambda x: 0 if x < thr_exp else 1)
        np.fill_diagonal(exp_matrix.values, 0)

        seq = pd.read_csv(os.path.join(
            './Data/simmatrix/seqsim_matrix.csv'), sep='\t', index_col=0)
        seq_matrix = seq.applymap(lambda x: 0 if x < thr_seq else 1)
        np.fill_diagonal(seq_matrix.values, 0)

        networklist = []
        for matrix in [go_matrix, exp_matrix, seq_matrix, path_matrix]:
            temp = pd.DataFrame(index=final_gene_node, columns=final_gene_node)
            network = temp.combine_first(matrix)
            network.fillna(0, inplace=True)
            network_adj = network[final_gene_node].loc[final_gene_node]
            networklist.append(network_adj)
            print('The shape of network_adj:', network_adj.shape)

        # Save the processed graph data and omic data
        ppi.to_csv(os.path.join(self.graph_path + '_ppi.tsv'),
                   sep='\t', index=False, compression='gzip')
        networklist[0].to_csv(os.path.join(
            self.graph_path, self.ppi_type + '_' + str(thr_go) + '_go.tsv'), sep='\t')
        networklist[1].to_csv(os.path.join(
            self.graph_path, self.ppi_type + '_' + str(thr_exp) + '_exp.tsv'), sep='\t')
        networklist[2].to_csv(os.path.join(
            self.graph_path, self.ppi_type + '_' + str(thr_seq) + '_seq.tsv'), sep='\t')
        networklist[3].to_csv(os.path.join(
            self.graph_path, self.ppi_type + '_' + str(thr_path) + '_path.tsv'), sep='\t')

        return ppi, networklist[0], networklist[1], networklist[2], networklist[3]

    def pyg_node2vec_embedding(self, edge_index, graph_type, node2vec_p, node2vec_q, epochs):

        model = Node2Vec(edge_index, embedding_dim=16, walk_length=80,
                         context_size=5, walks_per_node=10,
                         num_negative_samples=1, p=node2vec_p, q=node2vec_q, sparse=True).cuda()

        loader = model.loader(batch_size=128, shuffle=True, num_workers=1)
        optimizer = torch.optim.SparseAdam(
            list(model.parameters()), lr=0.001)
        best_loss = None
        for epoch in range(1, epochs+1):
            model.train()
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.cuda(), neg_rw.cuda())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch: {epoch:02d}, Loss: {total_loss/len(loader):.4f}')

            if best_loss is None:
                best_loss = total_loss/len(loader)
            else:
                decay = best_loss - total_loss/len(loader)
                print(
                    f'best_loss{best_loss}, loss{total_loss/len(loader)} decay{decay}')
                best_loss = total_loss/len(loader)
                if decay < 0.001:
                    print(f'The node2vec train stopped at epoch {epoch}!')
                    break

        model.eval()
        z = model()

        # scaler = preprocessing.MinMaxScaler()
        # z_norm = scaler.fit_transform(np.abs(z.cpu().detach()))

        torch.save(z, os.path.join(self.graph_path + '_' +
                   graph_type + '_node2vec_embedding.pkl'))
        print(f'The shape of node2vec embedding is {z.shape}')

        return z

    def load_featured_graph(self, network, omicfeature):

        omics_feature_vector = sp.csr_matrix(omicfeature, dtype=np.float32)
        omics_feature_vector = torch.FloatTensor(
            np.array(omics_feature_vector.todense()))
        print(
            f'The shape of omics_feature_vector:{omics_feature_vector.shape}')

        if network.shape[0] == network.shape[1]:
            G = nx.from_pandas_adjacency(network)
        else:
            G = nx.from_pandas_edgelist(network)

        G_adj = nx.convert_node_labels_to_integers(
            G, ordering='sorted', label_attribute='label')

        print(f'If the graph is connected graph: {nx.is_connected(G_adj)}')
        print(
            f'The number of connected components: {nx.number_connected_components(G_adj)}')

        graph = from_networkx(G_adj)
        assert graph.is_undirected() == True
        print(f'The edge index is {graph.edge_index}')

        graph.x = omics_feature_vector

        return graph
