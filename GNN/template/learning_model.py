import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl


# 自定义 Focal Loss
# 适用于不均衡样本
class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = 2 # 调制因子，降低容易识别样本的权重
        self.alpha = 0.75 # 降低 数量较多样本所占的权重 不行试下0.75

    # y_pred 二维tensor, y_real 一维LongTensor
    # y_real 相当于是index
    # [[0.4, 0.6], [0.7, 0.3]] <==>  [0, 1]
    # def forward(self, y_pred, y_real):
    #     y_real = y_real.type_as(y_pred)
    #     bce = nn.BCELoss(reduction="none")(y_pred, y_real)
    #     pt = y_pred * y_real +  (1 - y_pred) * (1 - y_real)
    #     alpha_factor = self.alpha * y_real + (1 - self.alpha) * (1 - y_real)
    #     modulating_factor = pt.pow(self.gamma)
    #     loss = torch.mean(alpha_factor * modulating_factor * bce)
        
    #     return loss
    def forward(self, pred, target):
        target = target.type_as(pred)
        pt = (1 - pred) * target + pred * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) *
                        (1 - target)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none') * focal_weight
        loss = torch.mean(loss)
        return loss


# 试试 隐藏层的特征数为 100？
class FLModel(nn.Module):
    def __init__(self):
        super().__init__()
        # #硬除掉7个特征之后的
        # self.fc1 = nn.Linear(158, 80)
        # self.fc5 = nn.Linear(80, 2)

        #选出120个特征之后的
        self.fc1 = nn.Linear(150, 64)
        self.s1 = nn.ReLU()
        # self.fc2 = nn.Linear(64, 8)
        # self.s2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 2)
        self.s = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.s1(x)
        # x = self.fc2(x)
        # x = self.s2(x)
        x = self.fc3(x)
        # output = self.s(x)

        return x

# 针对有向图的 GAT层

class GraphAttentionLayer(nn.Module):
    
    def __init__(self, n_in_feat, n_out_feat, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.n_in_feat = n_in_feat
        self.n_out_feat = n_out_feat
        self.dropout = dropout # 是否随机去掉参数，防止过拟合
        self.alpha = alpha # leakyrelu激活的参数
        self.concat = concat # 当前层是否是最后一层
        self.leakyrelu = nn.LeakyReLU(self.alpha) # 定义激活函数

        # 线性变换矩阵参数
        self.W = nn.Parameter(torch.empty(size=(n_in_feat, n_out_feat)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # 感知参数 用于计算最终的attention
        self.a = nn.Parameter(torch.empty(size=(2 * n_out_feat, 1)))
        nn.init.xavier_uniform_(self.a, gain=1.414)

    def forward(self, h, adj):
        # (N, n_in) ==> (N, n_out)
        h_W = torch.mm(h, self.W)
        e = self.get_attention_mechanism_input(h_W)
        
        # 这个无穷小矩阵的作用是 将e中不相邻节点对应位置的值 清空
        near_zero_mat = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, near_zero_mat)
        # softmax过后，e^(-00) 近似等于 0
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_out = torch.matmul(attention, h_W)
        if self.concat:
            # 如果不是最后一层
            return F.elu(h_out)
        else:
            # 最后一层则直接输出
            return h_out

    
    # 获取 未归一化的 attention矩阵 e
    def get_attention_mechanism_input(self, h_W):
        h_W1 = torch.matmul(h_W, self.a[:self.n_out_feat, :])
        h_W2 = torch.matmul(h_W, self.a[self.n_out_feat: , :])
        # 利用torch的广播机制相加，即自动复制填充
        e = h_W1 + h_W2.T
        # 最后别忘激活函数 激活一下
        return self.leakyrelu(e)

# 直接调用的上层神经网络模型
# # n_feat=165, n_hid=100, n_class=2, dropout=0.6, alpha=0.2, nheads=8
class GAT(nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.n_feat = 165
        self.n_hid = 50
        self.n_class = 2
        self.dropout = 0
        self.alpha = 0.2
        self.nheads = 2

        self.attentions = [GraphAttentionLayer(self.n_feat, self.n_hid, dropout=self.dropout, alpha=self.alpha, concat=True) for _ in range(self.nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # 最后的分类层也可用 GAT层
        # nheads 决定多少个GAT层的输出 拼接在一起，所以总长度就是 h_hid * nheads
        self.out_layer = GraphAttentionLayer(self.n_hid * self.nheads, self.n_class, self.dropout, self.alpha, False)

    def forward(self, h, adj):
        x = F.dropout(h, self.dropout, training=self.training)
        # 不同head的结果直接横向拼接
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        # 最后一层输出并激活
        x = F.elu(self.out_layer(x, adj)) 
        return F.log_softmax(x, dim=1) # log_softmax速度变快，保持数值稳定


class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, activation  = 'relu'):
        super(GraphConvLayer, self).__init__()

        self.W = torch.nn.Parameter(torch.DoubleTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.W)
        
        self.set_act = False
        if activation == 'relu':
            self.activation = nn.ReLU()
            self.set_act = True
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim = 1)
            self.set_act = True
        else:
            self.set_act = False
            raise ValueError("activations supported are 'relu' and 'softmax'")
            
        
    def forward(self, H_in, A):
        # A must be an n x n matrix as it is an adjacency matrix
        # H is the input of the node embeddings, shape will n x in_features
        self.A = A
        self.H_in = H_in
        A_ = torch.add(self.A, torch.eye(self.A.shape[0]).double())
        # D_为度矩阵
        D_ = torch.diag(A_.sum(1))
        # since D_ is a diagonal matrix, 
        # its root will be the roots of the diagonal elements on the principle diagonal
        # since A is an adjacency matrix, we are only dealing with positive values 
        # all roots will be real
        D_root_inv = torch.inverse(torch.sqrt(D_))
        A_norm = torch.mm(torch.mm(D_root_inv, A_), D_root_inv)
        # shape of A_norm will be n x n
        H_out = torch.mm(torch.mm(A_norm, H_in.double()), self.W)
        # shape of H_out will be n x out_features
        
        if self.set_act:
            H_out = self.activation(H_out)
            
        return H_out

    
class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.in_features = 165
        self.hidden_features = 50
        self.n_class = 2

        
        self.gcl1 = GraphConvLayer(self.in_features, self.hidden_features, activation  = 'relu')
        
        self.gcl2 = GraphConvLayer(self.hidden_features, self.n_class, activation = 'softmax')
        
    def forward(self, X, A):
        out = self.gcl1(X, A)

        out = self.gcl2(out, A)
            
        return out

class GraphSAGE(nn.Module):
    def __init__(self):
        super(GraphSAGE, self).__init__()


# 下面是 使用DGL建的模型
from dgl.nn import GraphConv

class DGLGCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(DGLGCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

from dgl.nn import TAGConv
class TAGConvLayer(nn.Module):
    def __init__(self, in_feats, h_heats, num_class):
        super(TAGConvLayer, self).__init__()
        self.conv1 = TAGConv(in_feats, h_heats, k=1) # k 代表 hop跳数
        self.conv2 = TAGConv(h_heats, num_class, k=1)
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

from dgl.nn import GATConv
class GATConvLayer(nn.Module):
    def __init__(self, in_feats, h_feats, num_class):
        super(GATConvLayer, self).__init__()
        self.muti_head1 = 2
        self.muti_head2 = 1
        self.conv1 = GATConv(in_feats, h_feats, num_heads=self.muti_head1)
        self.conv2 = GATConv(self.muti_head1 * h_feats, num_class, num_heads=self.muti_head2)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = h.reshape(h.shape[0], -1)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = h.reshape(h.shape[0], -1)
        return h

from dgl.nn import SAGEConv
class SAGEConvLayer(nn.Module):
    def __init__(self, in_feats, h_heats, num_class):
        super(SAGEConvLayer, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_heats, aggregator_type='mean')
        self.conv2 = SAGEConv(h_heats, num_class, aggregator_type='mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

from dgl.nn import SGConv
class SGConvLayer(nn.Module):
    def __init__(self, in_feats, h_heats, num_class):
        super(SGConvLayer, self).__init__()
        self.conv1 = SGConv(in_feats, h_heats, k = 1)
        self.conv2 = SGConv(h_heats, num_class, k = 1)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


from dgl.nn import APPNPConv
class APPNPConvLayer(nn.Module):
    def __init__(self, in_feats, h_heats, num_class):
        super(APPNPConvLayer, self).__init__()
        self.conv1 = APPNPConv(k=3, alpha=0.5)
        self.linear1 = nn.Linear(in_feats, h_heats)
        self.linear2 = nn.Linear(h_heats, num_class)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.linear1(h)
        h = F.relu(h)
        h = self.linear2(h)
        return h

# from dgl.nn import ChebConv
# class ChebConvLayer(nn.Module):
#     def __init__(self, in_feats, h_heats, num_class):
#         super(ChebConvLayer, self).__init__()
#         self.conv1 = ChebConv(in_feats, h_heats, k = 1)
#         self.conv2 = ChebConv(h_heats, num_class, k = 1)

#     def forward(self, g, in_feat):
#         h = self.conv1(g, in_feat)
#         h = F.relu(h)
#         h = self.conv2(g, h)
#         return h


class BTCModel(nn.Module):
    def __init__(self):
        super(BTCModel, self).__init__()
        # self.fc = GATConvLayer(165, 80, 2)
        # self.fc2 = TAGConvLayer(50, 20, 2)
        self.fc = FLModel()


    def forward(self, X):
        X = self.fc(X)
        # X = F.relu(X)
        # X = self.fc2(g, X)
        # output = F.log_softmax(X, dim=1)
        output = torch.sigmoid(X)
        return output

