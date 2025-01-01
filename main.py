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
import pandas as pd
from modig import MODIG
from modig_graph import ModigGraph
from utils import *

cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    # 命令行参数设置
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
                        default=50,
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
                        default=150,
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
    parser.add_argument('-thr_seq', '--thr_seq', help='the threshold for gene sequence similarity (default: 0.8)',
                        dest='thr_seq',
                        default=0.8,
                        type=float
                        )
    parser.add_argument('-thr_exp', '--thr_exp', help='the threshold for tissue co-expression pattern (default: 0.8)',
                        dest='thr_exp',
                        default=0.8,
                        type=float
                        )
    parser.add_argument('-thr_path', '--thr_path', help='the threshold of gene pathway co-occurrence (default: 0.6)',
                        dest='thr_path',
                        default=0.6,
                        type=float
                        )
    parser.add_argument('-seed', '--seed', help='the random seed (default: 42)',
                        dest='seed',
                        default=42,
                        type=int
                        )
    args = parser.parse_args()
    return args


import os
import torch

# 假设以下这些导入的模块在代码其他地方有相应的定义和使用
from modig_graph import ModigGraph
import pandas as pd


def main(args):
    """
    main函数作为整个程序的主入口，主要负责进行一系列初始化操作、数据加载以及为后续的模型训练和评估做准备工作，
    包括设置随机种子、创建保存结果的目录、读取或生成相关的图数据和组学特征数据、构建特征图列表、准备交叉验证数据集等。

    参数:
    - args: 一个字典类型的参数，里面包含了程序运行所需的各种配置参数，例如随机种子（'seed'）、蛋白质 - 蛋白质相互作用类型（'ppi'）、
            用于保存结果的标题（'title'）、癌症类型（'cancer'），还有各种阈值参数（如'thr_go'、'thr_exp'、'thr_seq'、'thr_path'）等，
            这些参数将用于控制程序中不同部分的行为和操作。
    """
    # 设置随机种子，确保在每次运行程序时，如果涉及到随机操作（如数据打乱、初始化模型参数等），都能得到可复现的结果，
    # 具体的seed_torch函数应该是实现了针对PyTorch相关的随机种子设置功能（例如设置CPU和GPU上的随机种子等）
    seed_torch(args['seed'])
    # 通过拼接路径字符串的方式构建用于保存结果的文件路径，将'./Output'、args['ppi']和args['title']这几个部分组合起来，
    # 其中args['ppi']和args['title']通常是根据具体的运行配置来确定的子目录或文件名相关部分，这样可以灵活指定保存位置
    file_save_path = os.path.join('./Output', args['ppi'], args['title'])
    # 创建刚才构建好的文件保存路径对应的目录，如果目录不存在的话就创建它，确保后续保存文件时目录是存在的，避免出现保存失败的情况
    make_dir(file_save_path)

    # load data（开始加载数据部分）
    # 构建图数据所在的目录路径，将'./Data/graph'和args['ppi']拼接起来，这里args['ppi']可能是用于区分不同类型的图数据（比如不同的蛋白质相互作用类型对应的图）
    graph_path = os.path.join('./Data/graph', args['ppi'])
    # 判断刚才构建的图数据目录是否存在，如果不存在就创建该目录，为后续读取或生成图数据文件做准备
    if not os.path.exists(graph_path):
        os.makedirs(graph_path)

    # 实例化ModigGraph类，传入图数据所在的路径（graph_path）、蛋白质 - 蛋白质相互作用类型（args['ppi']）以及癌症类型（args['cancer']），
    # 这个类的实例化对象（modig_input）后续将用于调用相关方法来获取图数据、组学特征数据以及进行一些和图相关的处理操作，具体功能依赖于该类的定义
    modig_input = ModigGraph(graph_path, args['ppi'], args['cancer'])

    print('Network INFO')

    # 构建蛋白质 - 蛋白质相互作用网络（PPI）数据文件的路径，通过将图数据所在路径（graph_path）、特定的文件名后缀（'_ppi.tsv'）与args['ppi']拼接而成，
    # 后续可以根据这个路径去读取对应的PPI网络数据文件（假设文件是以制表符分隔的表格形式存储，且第一列作为索引列，这在后面的读取代码中有所体现）
    ppi_path = os.path.join(graph_path, args['ppi'] + '_ppi.tsv')
    # 构建基因本体（GO）网络数据文件的路径，同样基于图数据路径，添加包含阈值参数（args['thr_go']）的文件名后缀（'_go.tsv'）以及args['ppi']来确定具体文件位置，
    # 这样可以准确找到对应不同阈值设置下的GO网络数据文件，方便后续读取和处理
    go_path = os.path.join(
        graph_path, args['ppi'] + '_' + str(args['thr_go']) + '_go.tsv')
    # 构建表达量（EXP）网络数据文件的路径，原理与前面类似，文件名后缀中包含对应表达量数据的阈值参数（args['thr_exp']），用于定位到准确的EXP网络数据文件
    exp_path = os.path.join(
        graph_path, args['ppi'] + '_' + str(args['thr_exp']) + '_exp.tsv')
    # 构建序列（SEQ）网络数据文件的路径，文件名中带有与序列数据相关的阈值参数（args['thr_seq']），以便能正确读取对应的SEQ网络数据文件
    seq_path = os.path.join(
        graph_path, args['ppi'] + '_' + str(args['thr_seq']) + '_seq.tsv')
    # 构建通路（PATH）网络数据文件的路径，包含对应通路数据的阈值参数（args['thr_path']）在文件名后缀中，用于获取相应的PATH网络数据文件
    path_path = os.path.join(
        graph_path, args['ppi'] + '_' + str(args['thr_path']) + '_path.tsv')

    # 构建组学特征数据文件的路径，由图数据所在路径（graph_path）、特定的文件名后缀（'_omics.tsv'）与args['ppi']拼接而成，
    # 用于后续读取包含基因等组学特征信息的数据文件
    omic_path = os.path.join(graph_path, args['ppi'] + '_omics.tsv')

    # 判断上述构建的五个基因相似性网络数据文件（PPI、GO、EXP、SEQ、PATH）以及组学特征数据文件是否都存在，
    # 通过使用逻辑与（&）操作符连接多个文件存在性判断，只有当所有文件都存在时，才执行if分支内的代码
    if os.path.exists(ppi_path) & os.path.exists(go_path) & os.path.exists(exp_path) & os.path.exists(seq_path) & os.path.exists(path_path) & os.path.exists(omic_path):
        print('The five gene similarity profiles and omic feature already exist!')
        # 如果所有文件都存在，使用pandas库的read_csv函数分别读取这五个网络数据文件以及组学特征数据文件，
        # 并指定分隔符为制表符（'\t'），将第一列作为索引列（index_col=0），然后将读取到的数据分别存储到对应的变量中，方便后续使用
        ppi_network = pd.read_csv(ppi_path, sep='\t', index_col=0)
        go_network = pd.read_csv(go_path, sep='\t', index_col=0)
        exp_network = pd.read_csv(exp_path, sep='\t', index_col=0)
        seq_network = pd.read_csv(seq_path, sep='\t', index_col=0)
        path_network = pd.read_csv(path_path, sep='\t', index_col=0)
        omicsfeature = pd.read_csv(omic_path, sep='\t', index_col=0)
        # 获取组学特征数据文件的索引列表，并将其转换为列表形式，作为最终的基因节点列表，这个列表通常包含了数据中涉及的所有基因名称或标识，
        # 在后续处理中可能用于对基因相关数据的筛选、匹配等操作，确保不同数据在相同的基因集合上进行分析和处理
        final_gene_node = list(omicsfeature.index)

    else:
        # 如果上述文件有不存在的情况，则需要调用modig_input对象的相关方法来生成或获取相应的数据
        # 首先调用modig_input的get_node_omicfeature方法，获取组学特征数据（omicsfeature）以及最终的基因节点列表（final_gene_node），
        # 该方法内部应该实现了从原始数据中提取、处理并返回组学特征及对应基因节点信息的功能
        omicsfeature, final_gene_node = modig_input.get_node_omicfeature()
        # 接着调用modig_input的generate_graph方法，传入不同的阈值参数（args['thr_go']、args['thr_exp']、args['thr_seq']、args['thr_path']），
        # 用于生成五个基因相似性网络（PPI、GO、EXP、SEQ、PATH）的数据，返回值分别赋值给对应的变量，以便后续使用这些网络数据进行进一步的处理和分析
        ppi_network, go_network, exp_network, seq_network, path_network = modig_input.generate_graph(
            args['thr_go'], args['thr_exp'], args['thr_seq'], args['thr_path'])

    print("==========================================================")
    print('Network INFO')
    # 定义一个包含五个网络名称的元组，分别对应前面提到的五个基因相似性网络（PPI、GO、EXP、SEQ、PATH），
    # 这个元组可以用于在后续的循环或打印等操作中方便地标识不同的网络，增强代码的可读性和可维护性
    name_of_network = ['PPI', 'GO', 'EXP', 'SEQ', 'PATH'],
    graphlist = []
    # 通过enumerate函数同时获取索引（i）和对应的值（network），循环遍历由五个网络数据（ppi_network、go_network、exp_network、seq_network、path_network）组成的列表，
    # 这样在循环内部可以方便地知道当前处理的是第几个网络以及对应的网络数据本身，便于进行针对性的操作和记录
    for i, network in enumerate([ppi_network, go_network, exp_network, seq_network, path_network]):
        # 调用modig_input对象的load_featured_graph方法，传入当前的网络数据（network）以及组学特征数据（omicsfeature），
        # 该方法应该是根据网络数据和组学特征来构建对应的特征图（featured_graph），具体构建方式依赖于该方法的内部实现逻辑，
        # 例如可能涉及到将网络结构与节点的组学特征进行整合等操作，以得到适合后续模型处理的图表示形式
        featured_graph = modig_input.load_featured_graph(network, omicsfeature)
        # 打印当前网络的相关信息，包括网络的序号（i）、网络名称（name_of_network[i]）以及构建好的特征图（featured_graph），
        # 这样可以在控制台输出中清晰地看到每个网络的处理情况和对应的特征图结构等信息，便于调试和了解程序运行状态
        print(f' The graph: {featured_graph}')
        # 将构建好的每个特征图添加到graphlist列表中，这个列表后续可能会用于批量处理多个特征图，比如作为模型的输入数据等，
        # 将多个相关的数据整合到一个列表中方便统一管理和操作
        graphlist.append(featured_graph)

    # 获取graphlist列表中第一个特征图的节点特征维度，即特征矩阵的列数，存储在n_fdim变量中，
    # 这里假设所有特征图的节点特征维度是一致的（从代码中看只取了第一个图的特征维度作为后续通用的维度参考），
    # 这个维度信息通常在初始化模型时需要用到，用于指定模型输入层接收的特征数量
    n_fdim = graphlist[0].x.shape[1]  # n_gene = featured_gsn.x.shape[0]
    # 使用列表推导式将graphlist中的每个特征图数据转移到GPU上（如果GPU可用的话），通过调用每个图对象的cuda方法实现，
    # 将数据放在GPU上可以利用GPU的并行计算能力，加速后续模型训练和相关计算操作，提高程序运行效率，得到的graphlist_adj列表包含了在GPU上的特征图数据
    graphlist_adj = [graph.cuda() for graph in graphlist]
    # 调用modig_input对象的ten_fold_five_crs_validation方法，传入文件保存路径（file_save_path）作为参数，
    # 该方法应该是用于进行十折五次交叉验证相关的数据划分操作，返回值分别赋值给k_sets、idx_list、label_list三个变量，
    # 具体每个变量包含的数据结构和含义取决于该方法的内部实现，一般来说会涉及到训练集、验证集等不同数据集的掩码、索引以及标签信息等，
    # 这些数据将用于后续在交叉验证过程中准确地选取训练和验证数据进行模型的训练和评估操作
    k_sets, idx_list, label_list = modig_input.ten_fold_five_crs_validation(
        file_save_path)
    print("==========================================================")

    # def train(mask, label):
    #     model.train()
    #     optimizer.zero_grad()
        
    #     # 确保label和output在同一个设备上
    #     label = label.to(device)  # 将label转到设备上
    #     output = model(graphlist_adj)
        
    #     loss = F.binary_cross_entropy_with_logits(
    #         output[mask], label, pos_weight=torch.Tensor([2.7]).to(device))  # 设备一致

    #     print("Label device:", label.device)
    #     print("Output device before conversion:", output[mask].device)
    #     print("Type of label after .cpu():", type(label.cpu()))
    #     print("Type of output after conversion:", type(torch.sigmoid(output[mask]).cpu().detach().numpy()))
        
    #     # 计算准确率
    #     acc = metrics.accuracy_score(label.cpu(), np.round(
    #         torch.sigmoid(output[mask]).cpu().detach().numpy()))  # 确保转到CPU进行计算
        
    #     loss.backward()
    #     optimizer.step()
    
    #     del output
    #     return loss.item(), acc


    # @torch.no_grad()
    # def test(mask, label):
    #     model.eval()
    #     output = model(graphlist_adj)
        
    #     # 确保label和output在同一个设备上
    #     label = label.to(device)  # 将label转到设备上
    #     loss = F.binary_cross_entropy_with_logits(
    #         output[mask], label, pos_weight=torch.Tensor([2.7]).to(device))  # 设备一致
        
    #     # 计算准确率
    #     acc = metrics.accuracy_score(label.cpu(), np.round(
    #         torch.sigmoid(output[mask]).cpu().detach().numpy()))  # 转到CPU进行计算
        
    #     pred = torch.sigmoid(output[mask]).cpu().detach().numpy()
    #     auroc = metrics.roc_auc_score(label.to('cpu'), pred)
    #     pr, rec, _ = metrics.precision_recall_curve(label.to('cpu'), pred)
    #     aupr = metrics.auc(rec, pr)
    
    #     return pred, loss.item(), acc, auroc, aupr

    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
    from torchmetrics import Accuracy, AUROC, AveragePrecision

    # 初始化 PyTorch metrics
    accuracy_metric = Accuracy(task='binary').to(device)
    auroc_metric = AUROC(task='binary').to(device)
    aupr_metric = AveragePrecision(task='binary').to(device)
    
    def reset_metrics():
        accuracy_metric.reset()
        auroc_metric.reset()
        aupr_metric.reset()
    
    def train(mask, label):
        model.train()
        optimizer.zero_grad()
    
        # 确保label和output在同一个设备上
        label = label.to(device)  # 将label转到设备上
        output = model(graphlist_adj)
    
        loss = F.binary_cross_entropy_with_logits(
            output[mask], label, pos_weight=torch.Tensor([2.7]).to(device))  # 设备一致
    
        # 计算准确率
        pred = torch.round(torch.sigmoid(output[mask]))
        acc = accuracy_metric(pred, label).item()
    
        loss.backward()
        optimizer.step()
    
        del output
        return loss.item(), acc
    
    
    @torch.no_grad()
    def test(mask, label):
        model.eval()
        reset_metrics()  # 重置指标
    
        output = model(graphlist_adj)
    
        # 确保label和output在同一个设备上
        label = label.to(device)  # 将label转到设备上
        
        # 对于二元交叉熵损失，保持label为浮点数
        loss = F.binary_cross_entropy_with_logits(
            output[mask], label, pos_weight=torch.Tensor([2.7]).to(device))
    
        # 计算预测值
        pred = torch.sigmoid(output[mask])
        
        # 将label转换为长整型，用于度量指标计算
        label_int = label.long()
        
        # 计算准确率（使用整数标签）
        acc = accuracy_metric(pred, label_int).item()
        
        # 计算 AUROC 和 AUPR（使用整数标签）
        auroc = auroc_metric(pred, label_int).item()
        aupr = aupr_metric(pred, label_int).item()
    
        return pred.cpu().numpy(), loss.item(), acc, auroc, aupr
    

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
            model = torch.nn.DataParallel(model)
            model.cuda()
            torch.cuda.empty_cache()
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
