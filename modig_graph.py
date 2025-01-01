#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : modig_graph.py
# @Time      : 2022/01/01 22:25:04
# @Author    : Zhao-Wenny

import os
import random

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse import csr_matrix
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

    # 该方法的目标是获取基因节点列表，并根据蛋白质 - 蛋白质相互作用（PPI）数据构建一个最终的基因节点的邻接矩阵

    # 该方法的目标是获取基因节点列表，并根据蛋白质 - 蛋白质相互作用（PPI）数据构建一个最终的基因节点的邻接矩阵
    def get_node_genelist(self):
        print('Get gene list')
        # 设置随机种子
        random.seed(42)
        gene = pd.read_csv("./Data/simmatrix/gene_info_for_GOSemSim.csv")
        gene_list = list(set(gene['Symbol']))

        ppi = pd.read_csv(os.path.join('./Data/ppi', self.ppi_type + '.tsv'), sep='\t',
                          compression='gzip', encoding='utf8', usecols=['partner1', 'partner2'])
        ppi.columns = ['source', 'target']
        ppi = ppi[ppi['source'] != ppi['target']]
        ppi.dropna(inplace=True)

        # 对基因节点列表进行采样，随机选择三分之一的节点
        sampled_gene_list = random.sample(gene_list, len(gene_list) // 3)
        # 对PPI数据中的源节点和目标节点分别进行采样，只保留采样后的节点相关的边
        sampled_source = [node for node in ppi['source'] if node in sampled_gene_list]
        sampled_target = [node for node in ppi['target'] if node in sampled_gene_list]
        ppi = ppi[(ppi['source'].isin(sampled_source)) & (ppi['target'].isin(sampled_target))]

        final_gene_node = sorted(
            list(set(sampled_gene_list) | set(sampled_source) | set(sampled_target)))

        G = nx.from_pandas_edgelist(ppi)

        ppi_df = nx.to_pandas_adjacency(G)

        temp = pd.DataFrame(index=final_gene_node, columns=final_gene_node)
        ppi_adj = temp.combine_first(ppi_df)
        ppi_adj.fillna(0, inplace=True)
        ppi_final = ppi_adj[final_gene_node].loc[final_gene_node]

        return final_gene_node, ppi_final

    def get_node_omicfeature(self):
        """
        该函数用于获取节点的组学特征相关信息，主要操作包括从文件读取组学数据，
        根据基因节点列表对数据进行处理、筛选，最终将处理好的数据转换为特定格式的特征向量并返回，
        同时返回最终的基因节点列表。
        """
        # 获取经过处理后的最终基因节点列表以及一个未使用的返回值（用下划线表示忽略）
        final_gene_node, _ = self.get_node_genelist()

        # process the omic data（处理组学数据）
        # 从指定路径的文件中读取组学数据，文件以制表符'\t'作为分隔符，并且将第一列作为索引列（index_col=0）
        omics_file = pd.read_csv(
            './Data/feature/biological_features.csv', sep='\t', index_col=0)

        # 将omics_file的索引（代表基因节点）与final_gene_node合并去重后进行排序，得到新的基因节点集合
        expendgene = sorted(list(set(omics_file.index) | set(final_gene_node)))
        # 创建一个以expendgene为索引、列与omics_file相同的空DataFrame，用于后续数据合并操作
        temp = pd.DataFrame(index=expendgene, columns=omics_file.columns)
        # 使用combine_first方法将omics_file的数据合并到temp中，优先保留omics_file中的数据
        omics_adj = temp.combine_first(omics_file)
        # 将合并后数据中的缺失值填充为0，直接在原数据上进行修改（inplace=True）
        omics_adj.fillna(0, inplace=True)
        # 从合并处理后的数据中筛选出final_gene_node对应的行数据，即只保留与最终基因节点相关的数据
        omics_adj = omics_adj.loc[final_gene_node]
        # 对筛选后的数据按照索引进行排序，同样是在原数据上进行操作（inplace=True）
        omics_adj.sort_index(inplace=True)

        # 根据不同的癌症类型（self.cancer_type）进行不同的数据筛选操作
        if self.cancer_type != 'pancan':
            """
            如果不是泛癌（'pancan'）类型，从omics_adj中筛选出列名包含当前癌症类型（self.cancer_type）的列数据，
            也就是只保留与当前特定癌症类型相关的数据列。
            """
            omics_data = omics_adj[omics_adj.columns[omics_adj.columns.str.contains(
                self.cancer_type)]]
        elif self.cancer_type == 'pancan':
            """
            如果是泛癌类型，选择16种特定的癌症项目名称
            """
            chosen_project = ['KIRC', 'BRCA', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD',
                              'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']
            # 对于每个选定的癌症项目，从omics_adj中筛选出列名包含对应癌症名称的列数据
            omics_temp = [omics_adj[omics_adj.columns[omics_adj.columns.str.contains(
                cancer)]] for cancer in chosen_project]
            # 将筛选出的多个与不同癌症相关的数据列沿着列方向（axis=1）进行拼接合并
            omics_data = pd.concat(omics_temp, axis=1)

        # 将筛选后的数据转换为scipy库中的压缩稀疏行格式（csr_matrix）的矩阵，指定数据类型为np.float32
        omics_feature_vector = sp.csr_matrix(omics_data, dtype=np.float32)
        # 再将稀疏矩阵转换为torch库中的FloatTensor类型，先将其转换为密集矩阵（todense）后再进行转换
        omics_feature_vector = torch.FloatTensor(
            np.array(omics_feature_vector.todense()))
        # 打印出组学特征向量的形状信息，方便查看数据维度情况
        print(
            f'The shape of omics_feature_vector:{omics_feature_vector.shape}')

        # 返回组学特征向量以及最终的基因节点列表，供其他函数或模块使用
        return omics_feature_vector, final_gene_node

    import pandas as pd
    import numpy as np

    def generate_graph(self, thr_go, thr_exp, thr_seq, thr_path):
        """
        generate_graph函数用于生成特定的图数据，具体是生成包含蛋白质-蛋白质相互作用（PPI）以及基于不同相似度阈值处理后的
        基因本体（GO）、表达量（EXP）、序列（SEQ）、通路（PATH）相关的图数据，操作包括读取原始数据、根据阈值进行二值化处理、
        构建图的邻接矩阵等，并将处理好的数据保存到文件（部分代码被注释掉，可能原本有保存功能），最后返回相关的图数据。
        """
        print('generate graph')
        # 获取最终的基因节点列表以及蛋白质-蛋白质相互作用（PPI）相关的数据（推测是邻接矩阵之类的数据结构）
        final_gene_node, ppi = self.get_node_genelist()

        # 读取通路相似度矩阵（pathsim_matrix.csv）数据，以制表符'\t'作为分隔符，并将第一列作为索引列（index_col=0）
        path = pd.read_csv(os.path.join(
            './Data/simmatrix/pathsim_matrix.csv'), sep='\t', index_col=0)
        # 根据给定的阈值thr_path对通路相似度矩阵进行二值化处理，小于阈值的元素设置为0，大于等于阈值的元素设置为1
        path_matrix = path.applymap(lambda x: 0 if x < thr_path else 1)
        # 将通路相似度矩阵的对角线元素（自身与自身的相似度，通常在构建图时设为0）设置为0
        np.fill_diagonal(path_matrix.values, 0)

        # 读取基因本体（GO）相似度矩阵（GOSemSim_matrix.csv）数据，同样以'\t'为分隔符，第一列作为索引列
        go = pd.read_csv('./Data/simmatrix/GOSemSim_matrix.csv',
                         sep='\t', index_col=0)
        # 按照给定的阈值thr_go对GO相似度矩阵进行二值化处理，小于阈值的设为0，大于等于阈值的设为1
        go_matrix = go.applymap(lambda x: 0 if x < thr_go else 1)
        # 将GO相似度矩阵的对角线元素设置为0
        np.fill_diagonal(go_matrix.values, 0)

        # 读取表达量相似度矩阵（expsim_matrix.csv）数据，以'\t'为分隔符，第一列作为索引列
        exp = pd.read_csv(os.path.join(
            './Data/simmatrix/expsim_matrix.csv'), sep='\t', index_col=0)
        # 根据阈值thr_exp对表达量相似度矩阵进行二值化处理，小于阈值的元素赋值为0，大于等于阈值的赋值为1
        exp_matrix = exp.applymap(lambda x: 0 if x < thr_exp else 1)
        # 将表达量相似度矩阵的对角线元素设置为0
        np.fill_diagonal(exp_matrix.values, 0)

        # 读取序列相似度矩阵（seqsim_rrbs_matrix.csv）数据，以'\t'为分隔符，第一列作为索引列
        seq = pd.read_csv(os.path.join(
            './Data/simmatrix/seqsim_rrbs_matrix.csv'), sep='\t', index_col=0)
        # 依据阈值thr_seq对序列相似度矩阵进行二值化处理，小于阈值的元素设为0，大于等于阈值的设为1
        seq_matrix = seq.applymap(lambda x: 0 if x < thr_seq else 1)
        # 将序列相似度矩阵的对角线元素设置为0
        np.fill_diagonal(seq_matrix.values, 0)

        networklist = []
        # 遍历经过二值化处理后的各个相似度矩阵（go_matrix、exp_matrix、seq_matrix、path_matrix）
        for matrix in [go_matrix, exp_matrix, seq_matrix, path_matrix]:
            # 创建一个以final_gene_node为索引和列名的空DataFrame，用于后续合并数据
            temp = pd.DataFrame(index=final_gene_node, columns=final_gene_node)
            # 将当前的相似度矩阵与空DataFrame进行合并，优先保留当前矩阵中的数据（使用combine_first方法）
            network = temp.combine_first(matrix)
            # 将合并后的数据中可能存在的缺失值填充为0，直接在原数据上进行修改（inplace=True）
            network.fillna(0, inplace=True)
            # 从合并处理后的数据中筛选出final_gene_node对应的行和列数据，即获取最终的邻接矩阵形式的数据
            network_adj = network[final_gene_node].loc[final_gene_node]
            # 将处理好的邻接矩阵添加到networklist列表中，用于后续保存或返回等操作
            networklist.append(network_adj)
            print('The shape of network_adj:', network_adj.shape)

        # Save the processed graph data and omic data（保存处理好的图数据和组学数据，以下部分代码被注释掉，可能原本有完整的保存逻辑）
        # 将ppi数据保存为以'_ppi.tsv'为后缀的文件，文件以'\t'为分隔符，不保存索引列（index=False），并使用gzip压缩
        # ppi.to_csv(os.path.join(self.graph_path + '_ppi.tsv'),
        #            sep='\t', index=False, compression='gzip')
        # 将ppi数据保存为以'_ppi.tsv'为后缀的文件，文件以'\t'为分隔符，这里使用self.ppi_type来构建文件名的一部分（具体含义需看代码其他部分）
        ppi.to_csv(os.path.join(
            self.graph_path, self.ppi_type + '_ppi.tsv'), sep='\t')
        # 将处理后的GO相关的邻接矩阵数据保存为以'_go.tsv'为后缀的文件，文件名中包含self.ppi_type和阈值thr_go信息，以'\t'为分隔符
        networklist[0].to_csv(os.path.join(
            self.graph_path, self.ppi_type + '_' + str(thr_go) + '_go.tsv'), sep='\t')
        # 将处理后的EXP相关的邻接矩阵数据保存为以'_exp.tsv'为后缀的文件，文件名中包含self.ppi_type和阈值thr_exp信息，以'\t'为分隔符
        networklist[1].to_csv(os.path.join(
            self.graph_path, self.ppi_type + '_' + str(thr_exp) + '_exp.tsv'), sep='\t')
        # 将处理后的SEQ相关的邻接矩阵数据保存为以'_seq.tsv'为后缀的文件，文件名中包含self.ppi_type和阈值thr_seq信息，以'\t'为分隔符
        networklist[2].to_csv(os.path.join(
            self.graph_path, self.ppi_type + '_' + str(thr_seq) + '_seq.tsv'), sep='\t')
        # 将处理后的PATH相关的邻接矩阵数据保存为以'_path.tsv'为后缀的文件，文件名中包含self.ppi_type和阈值thr_path信息，以'\t'为分隔符
        networklist[3].to_csv(os.path.join(
            self.graph_path, self.ppi_type + '_' + str(thr_path) + '_path.tsv'), sep='\t')

        # 返回ppi数据以及处理后的GO、EXP、SEQ、PATH相关的邻接矩阵数据，供其他函数或模块使用
        return ppi, networklist[0], networklist[1], networklist[2], networklist[3]
    # 这个函数的主要目的是利用
    # Node2Vec
    # 算法对给定的图结构（通过边索引
    # edge_index
    # 表示）进行节点嵌入（Node Embedding）学习，在训练过程中不断优化模型以减小损失，当满足一定条件时停止训练，最后将学习到的节点嵌入结果保存为文件，并返回该节点嵌入向量。节点嵌入旨在将图中的节点表示为低维向量，使得具有相似结构特征或语义信息的节点在向量空间中距离相近，方便后续用于图分析、节点分类等任务。
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

    # 该函数的主要作用是加载带有特征的图数据，将输入的组学特征数据（omicfeature）进行格式转换，同时根据提供的网络数据（network）构建图结构（使用
    # networkx
    # 库），对图结构进行一些必要的处理和属性检查，最后将组学特征数据关联到图结构上，返回处理好的图对象，以便后续用于图相关的分析或其他任务，比如图神经网络中的操作等。
    def load_featured_graph(self, network, omicfeature):
        network = network.applymap(lambda x: float(x) if isinstance(x, (int, float)) else x)
        omics_feature_vector = sp.csr_matrix(omicfeature, dtype=np.float32)

        omics_feature_vector = torch.FloatTensor(
            np.array(omics_feature_vector.todense()))  #后添加
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
