#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : model.py
# @Time      : 2022/01/01 22:15:22
# @Author    : Zhao-Wenny


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATConv


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)

        return (beta * z).sum(1), beta


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, mask=None):
        u = torch.matmul(q, k.transpose(-2, -1))
        u = u / self.scale

        if mask is not None:
            u = u.masked_fill(mask, -np.inf)

        attn = self.softmax(u)
        output = torch.matmul(attn, v)

        return attn, output


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, n_head, d_k_, d_v_, d_k, d_v, d_o):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.fc_q = nn.Linear(d_k_, n_head * d_k)
        self.fc_k = nn.Linear(d_k_, n_head * d_k)
        self.fc_v = nn.Linear(d_v_, n_head * d_v)

        self.attention = ScaledDotProductAttention(scale=np.power(d_k, 0.5))

        self.fc_o = nn.Linear(n_head * d_v, d_o)

    def forward(self, q, k, v, mask=None):

        n_head, d_q, d_k, d_v = self.n_head, self.d_k, self.d_k, self.d_v

        n_q, d_q_ = q.size()
        n_k, d_k_ = k.size()
        n_v, d_v_ = v.size()

        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)
        q = q.view(n_q, n_head, d_q).permute(
            1, 0, 2).contiguous().view(-1, n_q, d_q)
        k = k.view(n_k, n_head, d_k).permute(
            1, 0, 2).contiguous().view(-1, n_k, d_k)
        v = v.view(n_v, n_head, d_v).permute(
            1, 0, 2).contiguous().view(-1, n_v, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
        attn, output = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, n_q, d_v).permute(
            0, 1, 2).contiguous().view(n_q, -1)
        output = self.fc_o(output)

        return attn, output


class SelfAttention(nn.Module):
    """ Self-Attention """

    def __init__(self, n_head, d_k, d_v, d_x, d_o):
        super().__init__()
        self.wq = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wk = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wv = nn.Parameter(torch.Tensor(d_x, d_v))

        self.mha = MultiHeadAttention(
            n_head=n_head, d_k_=d_k, d_v_=d_v, d_k=d_k, d_v=d_v, d_o=d_o)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / np.power(param.size(-1), 0.5)
            param.data.uniform_(-stdv, stdv)

    def forward(self, x, mask=None):
        q = torch.matmul(x, self.wq)
        k = torch.matmul(x, self.wk)
        v = torch.matmul(x, self.wv)

        attn, output = self.mha(q, k, v, mask=mask)

        return output


class Gat_En(nn.Module):
    def __init__(self, nfeat, hidden_size, out, dropout):
        super(Gat_En, self).__init__()
        self.gat1 = GATConv(nfeat, hidden_size, heads=3, dropout=dropout)
        self.gat2 = GATConv(3*hidden_size, out, heads=1,
                            concat=True, dropout=dropout)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.gat2(x, edge_index))

        return x


class MODIG(nn.Module):
    def __init__(self, nfeat, hidden_size1, hidden_size2, dropout):
        super(MODIG, self).__init__()

        self.view_GNN = Gat_En(nfeat, hidden_size1, hidden_size2, dropout)

        self.self_attn = SelfAttention(
            n_head=1, d_k=64, d_v=32, d_x=hidden_size2, d_o=hidden_size2)
        self.attn = Attention(hidden_size2)

        self.MLP = nn.Linear(hidden_size2, 1)

        self.dropout = dropout

    def forward(self, graphs):

        embs = []
        for i in range(len(graphs)):
            emb = self.view_GNN(graphs[i])
            embs.append(emb)

        fused_outs = []
        for emb in embs:
            outs = self.self_attn(emb)
            fused_outs.append(outs)

        alpha = 0.6
        embs2 = []
        for i in range(len(embs)):
            emb2 = alpha * fused_outs[i] + (1 - alpha) * embs[i]
            embs2.append(emb2)

        emb_f, atts = self.attn(torch.stack(embs2, dim=1))

        output = self.MLP(emb_f)

        return output
