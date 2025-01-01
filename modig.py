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


# 定义Attention类，继承自nn.Module，通常用于实现注意力机制相关功能
class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        """
        类的初始化函数，用于定义网络层结构

        参数:
        - in_size: 输入数据的特征维度大小，即输入张量最后一维的大小，表示每个元素的特征数量
        - hidden_size: 隐藏层维度大小，默认为16，用于中间线性变换时调整特征维度，可根据实际情况调整
        """
        super(Attention, self).__init__()

        # 定义一个顺序容器（Sequential），包含多个神经网络层，按顺序依次执行
        self.project = nn.Sequential(
            # 第一个线性层，将输入维度为in_size的特征映射到维度为hidden_size的特征空间
            nn.Linear(in_size, hidden_size),
            # 使用Tanh激活函数对线性变换后的结果进行非线性变换，有助于增加模型的表达能力
            nn.Tanh(),
            # 第二个线性层，将维度为hidden_size的特征进一步映射到维度为1的空间，且不使用偏置（bias=False）
            # 这样最终输出的结果可用于后续计算注意力权重，维度为1便于后续进行softmax等操作
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        """
        前向传播函数，定义了数据在模型中的正向流动计算过程

        参数:
        - z: 输入张量，形状通常为[batch_size, sequence_length, in_size]，batch_size表示批次大小，
             sequence_length表示序列长度（如果是序列相关的数据），in_size为特征维度大小
        """
        # 将输入张量z传入定义好的project模块，得到中间计算结果w，此时w的形状通常为[batch_size, sequence_length, 1]
        w = self.project(z)
        # 对w在维度1（通常对应序列长度维度）上进行softmax操作，将每个位置的数值转换为表示注意力权重的概率分布，
        # 使得同一批次中每个序列位置对应的权重之和为1，beta的形状与w相同，为[batch_size, sequence_length, 1]
        beta = torch.softmax(w, dim=1)

        # 将beta（注意力权重）与输入张量z按元素相乘，并在维度1上求和，相当于根据注意力权重对z进行加权求和，
        # 返回的结果维度为[batch_size, in_size]，同时返回beta（注意力权重），方便后续可能的分析等操作
        return (beta * z).sum(1), beta


# 定义ScaledDotProductAttention类，继承自nn.Module，实现了缩放点积注意力机制
class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale):
        """
        初始化函数，用于初始化缩放点积注意力机制所需的参数和模块

        参数:
        - scale: 缩放因子，用于缩放点积计算的结果，避免数值过大或过小，影响后续计算（如softmax操作）
        """
        super().__init__()
        self.scale = scale
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, mask=None):
        """
        前向传播函数，定义了缩放点积注意力机制的计算流程

        参数:
        - q: 查询张量（query），形状通常为[batch_size, num_heads, sequence_length_q, head_dim]，
             其中num_heads表示多头注意力中的头数，head_dim表示每个头对应的维度大小，sequence_length_q表示查询序列长度
        - k: 键张量（key），形状通常为[batch_size, num_heads, sequence_length_k, head_dim]，sequence_length_k表示键序列长度
        - v: 值张量（value），形状通常为[batch_size, num_heads, sequence_length_v, head_dim]，sequence_length_v表示值序列长度
        - mask: 可选参数，掩码张量，形状通常与注意力分数矩阵（u）的形状匹配，用于在计算注意力时屏蔽某些位置，例如在处理变长序列时，
                对填充位置进行屏蔽，使其不参与注意力计算。如果不需要屏蔽，则为None
        """
        # 计算查询张量q与键张量k的转置（交换最后两个维度）的矩阵乘法，得到注意力分数矩阵u，
        # 形状为[batch_size, num_heads, sequence_length_q, sequence_length_k]，点积操作衡量了q与k之间的相关性
        u = torch.matmul(q, k.transpose(-2, -1))
        # 使用缩放因子self.scale对注意力分数矩阵u进行缩放，避免数值问题，这是缩放点积注意力机制的关键步骤之一
        u = u / self.scale

        # 如果传入了掩码张量mask，将注意力分数矩阵u中对应掩码位置的元素填充为负无穷（-np.inf），
        # 这样在后续进行softmax操作时，这些位置的概率会趋近于0，从而达到屏蔽的效果
        if mask is not None:
            u = u.masked_fill(mask, -np.inf)

        # 对经过缩放和（可能的）掩码处理后的注意力分数矩阵u进行softmax操作，在维度1（通常对应序列长度维度）上计算概率分布，
        # 得到注意力权重矩阵attn，形状与u相同，即[batch_size, num_heads, sequence_length_q, sequence_length_k]
        attn = self.softmax(u)
        # 将注意力权重矩阵attn与值张量v进行矩阵乘法，实现根据注意力权重对值进行加权求和，
        # 得到最终的输出output，形状通常为[batch_size, num_heads, sequence_length_q, value_dim]
        output = torch.matmul(attn, v)

        # 返回注意力权重矩阵attn以及最终的输出output，方便外部使用，例如可以查看注意力权重分布情况，
        # 或者将output用于后续的模型计算中
        return attn, output




# 定义MultiHeadAttention类，继承自nn.Module，实现多头注意力机制
class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, n_head, d_k_, d_v_, d_k, d_v, d_o):
        """
        初始化函数，用于定义多头注意力机制所需的各个层及参数

        参数:
        - n_head: 头的数量，表示同时并行计算的注意力头数，多头注意力可以从不同角度捕捉信息，增强模型表达能力
        - d_k_: 输入查询（query）特征的维度大小，即输入张量最后一维的大小，表示每个查询元素的原始特征数量
        - d_v_: 输入值（value）特征的维度大小，即输入张量最后一维的大小，表示每个值元素的原始特征数量
        - d_k: 经过线性变换后每个头对应的查询（query）特征维度大小，每个头会将输入特征映射到这个维度上进行注意力计算
        - d_v: 经过线性变换后每个头对应的 值（value）特征维度大小，每个头会将输入特征映射到这个维度上进行注意力计算
        - d_o: 多头注意力机制最终输出的特征维度大小，经过多头计算后会将结果融合并映射到这个维度上
        """
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # 定义线性变换层fc_q，用于将输入的查询特征维度从d_k_转换为n_head * d_k，
        # 目的是为每个头生成对应的查询特征表示，以便后续进行多头注意力计算
        self.fc_q = nn.Linear(d_k_, n_head * d_k)
        # 定义线性变换层fc_k，用于将输入的键（key）特征维度从d_k_转换为n_head * d_k，
        # 为每个头生成对应的键特征表示，用于和查询特征进行匹配计算注意力分数
        self.fc_k = nn.Linear(d_k_, n_head * d_k)
        # 定义线性变换层fc_v，用于将输入的值（value）特征维度从d_v_转换为n_head * d_v，
        # 为每个头生成对应的值特征表示，后续根据注意力分数对其进行加权求和
        self.fc_v = nn.Linear(d_v_, n_head * d_v)

        # 实例化ScaledDotProductAttention类，传入缩放因子scale=np.power(d_k, 0.5)，
        # 用于计算每个头的缩放点积注意力，它会根据查询、键和值计算注意力权重并进行加权求和
        self.attention = ScaledDotProductAttention(scale=np.power(d_k, 0.5))

        # 定义线性变换层fc_o，用于将多头注意力计算后的结果（维度为n_head * d_v）融合并映射到最终输出维度d_o上
        self.fc_o = nn.Linear(n_head * d_v, d_o)

    def forward(self, q, k, v, mask=None):
        """
        前向传播函数，定义了多头注意力机制的数据流动和计算过程

        参数:
        - q: 查询张量（query），形状通常为[batch_size, sequence_length_q, d_k_]，batch_size表示批次大小，
             sequence_length_q表示查询序列长度，d_k_为原始查询特征维度
        - k: 键张量（key），形状通常为[batch_size, sequence_length_k, d_k_]，sequence_length_k表示键序列长度，
             其原始特征维度与查询的原始特征维度一致，便于进行点积计算相关性
        - v: 值张量（value），形状通常为[batch_size, sequence_length_v, d_v_]，sequence_length_v表示值序列长度，
             d_v_为原始值特征维度
        - mask: 可选参数，掩码张量，形状通常根据具体应用场景而定（例如与注意力分数矩阵维度匹配），用于在计算注意力时屏蔽某些位置，
                如果不需要屏蔽则为None，常用于处理变长序列等情况，避免填充位置参与注意力计算影响结果
        """
        n_head, d_q, d_k, d_v = self.n_head, self.d_k, self.d_k, self.d_v

        n_q, d_q_ = q.size()
        n_k, d_k_ = k.size()
        n_v, d_v_ = v.size()

        # 使用线性变换层fc_q对查询张量q进行线性变换，将其特征维度从d_q_转换为n_head * d_k，
        # 为后续每个头的计算准备特征表示
        q = self.fc_q(q)
        # 使用线性变换层fc_k对键张量k进行线性变换，将其特征维度从d_k_转换为n_head * d_k
        k = self.fc_k(k)
        # 使用线性变换层fc_v对值张量v进行线性变换，将其特征维度从d_v_转换为n_head * d_v
        v = self.fc_v(v)

        # 对变换后的查询张量q进行形状调整，先将其视图（view）调整为[batch_size, n_head, d_q]的形式，
        # 然后通过permute操作交换维度顺序为[ n_head, batch_size, d_q]，使得头的维度在最前面，方便后续并行计算，
        # 最后使用contiguous方法确保内存连续，并再次调整视图为[-1, batch_size, d_q]的形式，方便后续计算
        q = q.view(n_q, n_head, d_q).permute(1, 0, 2).contiguous().view(-1, n_q, d_q)
        # 对键张量k进行类似的形状调整操作，将其转换为适合多头注意力计算的形状
        k = k.view(n_k, n_head, d_k).permute(1, 0, 2).contiguous().view(-1, n_k, d_k)
        # 对值张量v进行类似的形状调整操作，将其转换为适合多头注意力计算的形状
        v = v.view(n_v, n_head, d_v).permute(1, 0, 2).contiguous().view(-1, n_v, d_v)

        if mask is not None:
            # 如果传入了掩码张量mask，将其在头的维度上进行重复（repeat），使得每个头都能应用相同的掩码，
            # 保证掩码的维度与经过形状调整后的查询、键、值张量的维度匹配，以便在注意力计算中正确屏蔽相应位置
            mask = mask.repeat(n_head, 1, 1)

        # 调用ScaledDotProductAttention实例的forward方法，传入调整形状后的查询、键、值张量以及可能的掩码张量，
        # 计算得到每个头的注意力权重矩阵attn以及经过加权求和后的输出output
        attn, output = self.attention(q, k, v, mask=mask)

        # 对多头注意力的输出output进行形状调整，先将其视图调整为[n_head, batch_size, d_v]的形式，
        # 然后交换维度顺序为[batch_size, n_head, d_v]，再通过contiguous方法确保内存连续，
        # 最后调整视图为[batch_size, -1]的形式，将多头的结果融合为一个维度，准备传入后续的线性变换层
        output = output.view(n_head, n_q, d_v).permute(0, 1, 2).contiguous().view(n_q, -1)

        # 使用线性变换层fc_o对融合后的输出进行线性变换，将其特征维度从n_head * d_v映射到最终输出维度d_o上，
        # 得到多头注意力机制最终的输出结果
        output = self.fc_o(output)

        # 返回每个头的注意力权重矩阵attn以及最终的输出output，attn可用于分析注意力分布情况，
        # output可用于后续的模型计算，例如传递给下一层神经网络等
        return attn, output


# 定义SelfAttention类，继承自nn.Module，实现自注意力机制，通常用于序列内部元素之间的关联计算
class SelfAttention(nn.Module):
    """ Self-Attention """

    def __init__(self, n_head, d_k, d_v, d_x, d_o):
        """
        初始化函数，用于定义自注意力机制所需的参数和层结构

        参数:
        - n_head: 头的数量，与多头注意力机制中的头数含义相同，用于并行从不同角度捕捉序列内部元素之间的关系
        - d_k: 每个头对应的查询（query）和键（key）特征维度大小，经过线性变换后将输入特征映射到这个维度上进行注意力计算
        - d_v: 每个头对应的 值（value）特征维度大小，经过线性变换后将输入特征映射到这个维度上进行加权求和等操作
        - d_x: 输入张量的特征维度大小，即输入序列中每个元素的原始特征数量，用于确定线性变换的输入维度
        - d_o: 自注意力机制最终输出的特征维度大小，经过自注意力计算后将结果映射到这个维度上，以便后续与其他层对接
        """
        super().__init__()

        # 定义可学习的参数张量wq，用于将输入特征维度从d_x转换为d_k，作为查询向量的生成权重，
        # 参数初始化为随机值，后续会在训练过程中根据数据进行调整优化
        self.wq = nn.Parameter(torch.Tensor(d_x, d_k))
        # 定义可学习的参数张量wk，用于将输入特征维度从d_x转换为d_k，作为键向量的生成权重，
        # 与查询向量配合用于计算注意力分数
        self.wk = nn.Parameter(torch.Tensor(d_x, d_k))
        # 定义可学习的参数张量wv，用于将输入特征维度从d_x转换为d_v，作为值向量的生成权重，
        # 根据注意力分数对其进行加权求和得到最终输出
        self.wv = nn.Parameter(torch.Tensor(d_x, d_v))

        # 实例化MultiHeadAttention类，传入相应参数，利用已经定义好的多头注意力机制来实现自注意力功能，
        # 这里将输入特征同时作为查询、键和值进行注意力计算，捕捉序列内部元素间的关系
        self.mha = MultiHeadAttention(
            n_head=n_head, d_k_=d_k, d_v_=d_v, d_k=d_k, d_v=d_v, d_o=d_o)

        # 调用初始化参数的方法，对定义的可学习参数进行初始化，保证参数初始值在合理范围内，利于模型训练收敛
        self.init_parameters()

    def init_parameters(self):
        """
        初始化参数的方法，用于对类中定义的可学习参数进行合理的初始化操作

        原理是根据参数的最后一维大小（通常表示特征维度），计算一个合适的标准差（stdv），
        然后将参数的值均匀分布在[-stdv, stdv]区间内，这样可以避免参数初始值过大或过小，
        影响模型训练过程中的梯度更新等情况，有助于模型更好地收敛。
        """
        for param in self.parameters():
            stdv = 1. / np.power(param.size(-1), 0.5)
            param.data.uniform_(-stdv, stdv)

    def forward(self, x, mask=None):
        """
        前向传播函数，定义了自注意力机制的数据计算流程

        参数:
        - x: 输入张量，形状通常为[batch_size, sequence_length, d_x]，batch_size表示批次大小，
             sequence_length表示序列长度，d_x为输入序列中每个元素的原始特征数量
        - mask: 可选参数，掩码张量，与多头注意力机制中的掩码作用类似，用于屏蔽序列中某些位置，
                如果不需要屏蔽则为None，常用于处理变长序列等情况，避免填充位置参与注意力计算影响结果
        """
        # 使用输入张量x与查询权重张量wq进行矩阵乘法，生成查询向量q，其形状变为[batch_size, sequence_length, d_k]，
        # 这里的矩阵乘法是将输入特征映射到查询特征空间，以便后续计算注意力
        q = torch.matmul(x, self.wq)
        # 使用输入张量x与键权重张量wk进行矩阵乘法，生成键向量k，其形状变为[batch_size, sequence_length, d_k]，
        # 用于和查询向量配合计算注意力分数，衡量序列元素之间的相关性
        k = torch.matmul(x, self.wk)
        # 使用输入张量x与值权重张量wv进行矩阵乘法，生成值向量v，其形状变为[batch_size, sequence_length, d_v]，
        # 后续会根据注意力分数对这些值向量进行加权求和得到最终输出
        v = torch.matmul(x, self.wv)

        # 调用MultiHeadAttention实例的forward方法，传入生成的查询、键、值向量以及可能的掩码张量，
        # 利用多头注意力机制进行自注意力计算，得到注意力权重矩阵attn以及最终的输出output
        attn, output = self.mha(q, k, v, mask=mask)

        # 返回自注意力机制计算得到的最终输出output，该输出可用于后续的模型计算，例如传递给下一层神经网络，
        # 进行特征融合、分类等任务，而attn可用于分析序列元素之间的注意力分布情况
        return output




# 定义Gat_En类，继承自nn.Module，很可能是一个基于图注意力网络（GAT）的编码器类，用于处理图数据并提取特征
class Gat_En(nn.Module):
    def __init__(self, nfeat, hidden_size, out, dropout):
        """
        初始化函数，用于定义图注意力网络编码器的各层结构及相关参数

        参数:
        - nfeat: 输入特征的维度大小，即图中每个节点的特征向量维度，表示节点初始特征的数量
        - hidden_size: 隐藏层特征维度大小，是图注意力网络中中间层节点特征的维度，用于调整特征表示
        - out: 输出特征的维度大小，即经过图注意力网络处理后最终输出的节点特征维度，用于后续任务
        - dropout: 丢弃率（Dropout概率），用于在训练过程中随机丢弃部分神经元，防止过拟合，取值范围通常在0到1之间
        """
        super(Gat_En, self).__init__()

        # 定义第一个图注意力卷积层（GATConv），将输入特征维度nfeat转换为隐藏层特征维度hidden_size，
        # 设置头的数量为3（heads=3），意味着同时从3个不同的角度计算注意力权重并聚合信息，同时传入指定的丢弃率
        self.gat1 = GATConv(nfeat, hidden_size, heads=3, dropout=dropout)
        # 定义第二个图注意力卷积层（GATConv），将上一层输出的特征维度（3 * hidden_size，因为上一层有3个头）
        # 转换为最终输出的特征维度out，头的数量设置为1（heads=1），并且设置concat=True表示将多头输出的结果拼接起来，同样传入指定的丢弃率
        self.gat2 = GATConv(3 * hidden_size, out, heads=1,
                            concat=True, dropout=dropout)

        self.dropout = dropout

    def forward(self, data):
        """
        前向传播函数，定义了数据在图注意力网络编码器中的流动和计算过程

        参数:
        - data: 输入的数据对象，通常是包含图结构信息和节点特征信息的数据结构（比如在torch_geometric库中特定的数据格式），
                这里期望从中获取节点特征x和边索引edge_index用于图卷积操作
        """
        # 从输入数据对象中获取节点特征张量x和边索引张量edge_index，节点特征张量形状通常为[num_nodes, nfeat]，
        # 表示图中节点数量以及每个节点的特征维度；边索引张量通常是二维的，用于表示图中边的连接关系
        x, edge_index = data.x, data.edge_index
        # 根据设定的丢弃率dropout，对节点特征张量x进行随机丢弃操作（在训练阶段生效，根据self.training判断是否处于训练状态），
        # 有助于防止模型过拟合，丢弃操作会随机将部分元素置为0
        x = F.dropout(x, self.dropout, training=self.training)
        # 将经过丢弃操作后的节点特征x传入第一个图注意力卷积层gat1进行计算，使用ELU激活函数（F.elu）对输出结果进行非线性变换，
        # 得到经过第一层图注意力网络处理后的节点特征，激活函数可以增加模型的非线性表达能力
        x = F.elu(self.gat1(x, edge_index))
        # 再次对第一层输出的节点特征x进行随机丢弃操作，同样在训练阶段根据设定的丢弃率进行，进一步防止过拟合
        x = F.dropout(x, self.dropout, training=self.training)
        # 将经过第二次丢弃操作后的节点特征x传入第二个图注意力卷积层gat2进行计算，同样使用ELU激活函数对输出结果进行非线性变换，
        # 得到最终经过图注意力网络编码器处理后的节点特征x，该特征将作为输出返回，可用于后续的任务，比如分类、预测等
        x = F.elu(self.gat2(x, edge_index))

        return x


# 定义MODIG类，继承自nn.Module，可能是一个完整的模型类，整合了图注意力网络、注意力机制以及多层感知机等组件，用于特定的任务
class MODIG(nn.Module):
    def __init__(self, nfeat, hidden_size1, hidden_size2, dropout):
        """
        初始化函数，用于定义MODIG模型的各个组成部分及相关参数

        参数:
        - nfeat: 输入特征的维度大小，即模型最初接收的图中节点特征维度，与Gat_En类中的输入特征维度含义一致
        - hidden_size1: 第一个隐藏层的特征维度大小，用于图注意力网络编码器（Gat_En）中第一个隐藏层的维度设置，
                        可以理解为中间特征表示的一个过渡维度
        - hidden_size2: 第二个隐藏层的特征维度大小，是图注意力网络编码器输出以及后续注意力机制等操作所涉及的特征维度，
                        起到承上启下的作用，用于进一步调整和融合特征表示
        - dropout: 丢弃率（Dropout概率），在多个组件中用于防止过拟合，取值范围通常在0到1之间，整个模型使用统一的丢弃率设置
        """
        super(MODIG, self).__init__()

        # 实例化Gat_En类，创建一个图注意力网络编码器对象，传入相应的输入特征维度、隐藏层特征维度、输出特征维度以及丢弃率参数，
        # 用于对输入的图数据进行特征提取和转换，将图的节点特征从初始维度nfeat转换为维度hidden_size2的特征表示
        self.view_GNN = Gat_En(nfeat, hidden_size1, hidden_size2, dropout)

        # 实例化SelfAttention类，创建一个自注意力机制对象，设置头的数量为1（n_head=1），并指定查询、键、值的特征维度以及输入、输出特征维度，
        # 用于对经过图注意力网络编码器处理后的特征进行进一步的自注意力计算，捕捉特征内部的依赖关系，调整特征表示
        self.self_attn = SelfAttention(
            n_head=1, d_k=64, d_v=32, d_x=hidden_size2, d_o=hidden_size2)

        # 实例化Attention类，创建一个注意力机制对象，用于对经过特定处理后的特征进行注意力加权操作，进一步融合和筛选特征信息
        self.attn = Attention(hidden_size2)

        # 定义一个多层感知机（MLP）层，即一个简单的线性层，将经过前面处理后的特征维度从hidden_size2映射到维度1，
        # 通常可用于输出最终的预测结果（比如二分类任务中的类别概率等）
        self.MLP = nn.Linear(hidden_size2, 1)

        self.dropout = dropout

    def forward(self, graphs):
        """
        前向传播函数，定义了数据在MODIG模型中的完整处理流程，包括图特征提取、注意力机制应用以及最终结果输出等步骤

        参数:
        - graphs: 输入的图数据列表，每个元素是一个包含图结构和节点特征等信息的数据对象（类似前面Gat_En类中输入的data格式），
                  模型会对列表中的每个图分别进行处理，并整合结果进行后续操作
        """
        embs = []
        # 遍历输入的图数据列表graphs，对每个图调用view_GNN（图注意力网络编码器）进行特征提取，
        # 将得到的每个图的特征表示添加到embs列表中，用于后续的进一步处理和融合
        for i in range(len(graphs)):
            emb = self.view_GNN(graphs[i])
            embs.append(emb)

        fused_outs = []
        # 遍历经过图注意力网络编码器处理后得到的特征列表embs，对每个特征调用self_attn（自注意力机制）进行计算，
        # 将自注意力机制输出的结果添加到fused_outs列表中，进一步调整特征表示，捕捉特征内部的关联信息
        for emb in embs:
            outs = self.self_attn(emb)
            fused_outs.append(outs)

        alpha = 0.6
        embs2 = []
        # 对经过图注意力网络编码器处理后的特征embs和经过自注意力机制处理后的特征fused_outs进行加权融合操作，
        # 根据设定的权重alpha（这里alpha = 0.6），计算融合后的特征表示，并添加到embs2列表中，融合后的特征综合了两种不同处理方式得到的信息
        for i in range(len(embs)):
            emb2 = alpha * fused_outs[i] + (1 - alpha) * embs[i]
            embs2.append(emb2)

        # 将融合后的特征列表embs2转换为张量形式（通过torch.stack操作沿着指定维度dim=1堆叠），
        # 然后传入attn（注意力机制）进行注意力加权操作，得到加权后的特征表示emb_f以及对应的注意力权重atts，
        # 进一步筛选和突出重要的特征信息
        emb_f, atts = self.attn(torch.stack(embs2, dim=1))

        # 将经过注意力加权后的特征表示emb_f传入MLP（多层感知机）层进行线性映射，将特征维度从hidden_size2转换为维度1，
        # 得到最终的输出结果output，该结果可根据具体任务进行解读，比如在分类任务中可作为类别概率等
        output = self.MLP(emb_f)

        return output