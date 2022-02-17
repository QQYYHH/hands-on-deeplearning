import torch
import torch.nn.functional as F
from torch import nn
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from learning_model import FocalLoss

from preprocess import CompDataset
from preprocess import get_user_data

import dgl

# 本类用于 训练集 数据预处理 & 实际训练

class Worker(object):
    def __init__(self, user_idx):
        self.user_idx = user_idx
        self.data,self.edges = get_user_data(self.user_idx)  # The worker can only access its own data
        self.ps_info = {}

        self.txId_map = {} # 将过滤后的 txId离散化为 连续的整数值，便于求解邻接矩阵
        self.adj = None # 邻接矩阵
        self.remain_txId = [] # 保留下来的 txId

        self.train_loss = FocalLoss()

        # 下面加入 DGL需要用到的一些字段
        self.g = None

    # 本地数据集预处理 
    # 2 - unknown
    def preprocess_worker_data(self):
        # x = self.create_adjacency_matrix()
        # x = x.iloc[:, 2:]
        # 第一次过滤
        # 尝试使用所有数据集，进行图的半监督学习

        self.data = self.data[self.data['class']!=2]
        self.remain_txId = self.data['txId'].reset_index(drop=True)
        # xgboost筛选特征
        feature_name =['Local_feature_61', 'Local_feature_25', 'Aggregate_feature_47', 'Aggregate_feature_11', 'Aggregate_feature_62', 'Local_feature_58', 'Aggregate_feature_16', 'Aggregate_feature_24', 'Aggregate_feature_69', 'Aggregate_feature_17', 'Aggregate_feature_43', 'Local_feature_87', 'Aggregate_feature_23', 'Aggregate_feature_64', 'Aggregate_feature_31', 'Local_feature_32', 'Local_feature_46', 'Local_feature_44', 'Local_feature_80', 'Aggregate_feature_18', 'Local_feature_57', 'Local_feature_91', 'Local_feature_51', 'Local_feature_52', 'Aggregate_feature_56', 'Aggregate_feature_9', 'Aggregate_feature_63', 'Local_feature_34', 'Aggregate_feature_10', 'Aggregate_feature_40', 'Local_feature_30', 'Local_feature_11', 'Local_feature_37', 'Aggregate_feature_38', 'Local_feature_4', 'Aggregate_feature_66', 'Local_feature_16', 'Local_feature_28', 'Local_feature_60', 'Aggregate_feature_12', 'Local_feature_9', 'Aggregate_feature_42', 'Local_feature_35', 'Aggregate_feature_32', 'Local_feature_8', 'Aggregate_feature_8', 'Aggregate_feature_55', 'Aggregate_feature_29', 'Local_feature_55', 'Local_feature_83', 'Local_feature_45', 'Aggregate_feature_1', 'Aggregate_feature_14', 'Local_feature_43', 'Local_feature_65', 'Local_feature_73', 'Aggregate_feature_71', 'Local_feature_56', 'Aggregate_feature_20', 'Local_feature_1', 'Local_feature_66', 'Local_feature_63', 'Aggregate_feature_51', 'Local_feature_19', 'Local_feature_50', 'Aggregate_feature_35', 'Aggregate_feature_46', 'Local_feature_21', 'Local_feature_53', 'Local_feature_20', 'Aggregate_feature_41', 'Aggregate_feature_19', 'Local_feature_79', 'Local_feature_86', 'Local_feature_7', 'Aggregate_feature_44', 'Aggregate_feature_27', 'Local_feature_17', 'Local_feature_42', 'Aggregate_feature_67', 'Local_feature_93', 'Local_feature_10', 'Local_feature_36', 'Aggregate_feature_53', 'Aggregate_feature_34', 'Local_feature_69', 'Aggregate_feature_28', 'Aggregate_feature_48', 'Aggregate_feature_61', 'Local_feature_3', 'Aggregate_feature_52', 'Aggregate_feature_30', 'Aggregate_feature_15', 'Aggregate_feature_57', 'Local_feature_89', 'Local_feature_48', 'Local_feature_12', 'Local_feature_78', 'Local_feature_47', 'Local_feature_26', 'Aggregate_feature_49', 'Aggregate_feature_45', 'Local_feature_14', 'Local_feature_33', 'Aggregate_feature_3', 'Aggregate_feature_4', 'Local_feature_90', 'Aggregate_feature_54', 'Local_feature_23', 'Aggregate_feature_13', 'Local_feature_64', 'Aggregate_feature_58', 'Local_feature_72', 'Aggregate_feature_21', 'Aggregate_feature_6', 'Aggregate_feature_22', 'Local_feature_15', 'Aggregate_feature_68', 'Local_feature_13', 'Aggregate_feature_72', 'Local_feature_75', 'Aggregate_feature_25', 'Local_feature_18', 'Aggregate_feature_5', 'Aggregate_feature_50', 'Aggregate_feature_60', 'Aggregate_feature_36', 'Local_feature_22', 'Aggregate_feature_70', 'Local_feature_24', 'Local_feature_27', 'Local_feature_82', 'Local_feature_39', 'Local_feature_62', 'Local_feature_5', 'Aggregate_feature_33', 'Local_feature_29', 'Local_feature_38', 'Aggregate_feature_37', 'Local_feature_70', 'Local_feature_92', 'Aggregate_feature_7', 'Aggregate_feature_39', 'Local_feature_85', 'Aggregate_feature_2', 'Local_feature_81', 'Local_feature_88', 'Local_feature_49', 'Local_feature_74', 'Local_feature_54']
        x = self.data.loc[:, feature_name]

        # x = self.data.iloc[:, 2:]
        x = x.reset_index(drop=True)
        x = x.to_numpy().astype(np.float32)
        y = self.data['class']
        y = y.reset_index(drop=True)
        x[x == np.inf] = 1.
        x[np.isnan(x)] = 0.
        self.data = (x,y)
        # 建立 DGL图
        self.create_dgl_graph()


    def compute_mean_and_standard(self):
        x = self.data[0]
        x_mean = pd.DataFrame(x).mean().tolist()
        x_std = pd.DataFrame(x).std().tolist()
        n_sample_worker = len(x)
        return x_mean,x_std,n_sample_worker

    def feature_standard(self,agg_mean_all,agg_std_all):
        for j in range(0,len(agg_mean_all)):
            for i in range(0,len(self.data[0])):
                self.data[0][i][j] = (self.data[0][i][j]-agg_mean_all[j])/agg_std_all[j]


    # 新增建图函数 - 构建邻接矩阵
    def create_adjacency_matrix(self):
        # 对txId 离散化，映射为连续的 编号，便于求解邻接矩阵
        txId_list = self.remain_txId.values

        for index, txId in enumerate(txId_list):
            self.txId_map[txId] = index
        
        # 下面进行具体建图操作
        N = len(txId_list)
        self.adj = torch.zeros(N, N)
        txId1_list = self.edges['txId1'].values
        txId2_list = self.edges['txId2'].values
        for txId1, txId2 in zip(txId1_list, txId2_list):
            if txId1 in self.txId_map.keys() and txId2 in self.txId_map.keys():
                self.adj[self.txId_map[txId1], self.txId_map[txId2]] = 1


    # n_round并没有用到，后面可以考虑用下，充分利用每个节点的数据集
    def round_data(self, n_round, n_round_samples=-1):
        """Generate data for user of user_idx at round n_round.

        Args:
            n_round: int, round number
            n_round_samples: int, the number of samples this round
        """

        if n_round_samples == -1:
            return self.data

        n_samples = len(self.data[1])
        
        choices = np.random.choice(n_samples, min(n_samples, n_round_samples), replace=False)
        # 第二次过滤（随机挑选数据）txId也要筛选
        self.remain_txId = self.remain_txId.reindex(choices).reset_index(drop=True)
        return self.data[0][choices], self.data[1][choices]
    
    def receive_server_info(self, info): # receive info from PS
        self.ps_info = info
    
    def process_mean_round_train_acc(self): # process the "mean_round_train_acc" info from server
        mean_round_train_acc = self.ps_info["mean_round_train_acc"]
        # You can go on to do more processing if needed

    # some loss function need the shape of y is the same as X 
    # here wo do one-hot for y
    def cal_loss_need_one_hot(self, X, y):
        one_hot = torch.zeros(X.shape[0], X.shape[1]).scatter_(1, y.reshape(-1, 1), 1)
        output = self.train_loss(X, one_hot)
        return output


    # 训练集的具体训练过程
    # 没有梯度下降 进行梯度积累
    # 这里修改 loader算法，想想对图怎么mini-batch
    # 初步尝试，不进行 图的mini-batch，直接训练整个图
    def user_round_train(self, model, device, n_round,  batch_size, n_round_samples=-1, debug=False):

        # return self.dgl_train_graph(model, device, n_round,  batch_size, n_round_samples, debug)

        X,Y = self.round_data(n_round, n_round_samples)
        data = CompDataset(X=X, Y=Y)
        train_loader = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            shuffle=True,
        )

        model.train()
        correct = 0
        prediction = []
        real = []
        total_loss = 0
        model = model.to(device)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # import ipdb
            # ipdb.set_trace()
            # print(data.shape, target.shape)
            output = model(data)
            
            # negative log likehood loss
            # loss = F.nll_loss(output, target)

            # 试新的Loss
            loss = self.cal_loss_need_one_hot(output, target)

            total_loss += loss
            loss.backward()
            pred = output.argmax(
                dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            prediction.extend(pred.reshape(-1).tolist())
            real.extend(target.reshape(-1).tolist())

        # 计算F1 
        f1 = self.cal_f1(prediction, real)


        # 返回这一轮训练的样本总数 + 每个参数上累积的梯度
        grads = {'n_samples': data.shape[0], 'named_grads': {}}
        for name, param in model.named_parameters():
            grads['named_grads'][name] = param.grad.detach().cpu().numpy()
        
        worker_info = {}
        worker_info["train_acc"] = correct / len(train_loader.dataset)
        worker_info["loss"] = total_loss.data
        worker_info["f1"] = f1

        if debug:
            print('Training Loss: {:<10.2f}, accuracy: {:<8.2f}'.format(
                total_loss, 100. * correct / len(train_loader.dataset)))

        return grads, worker_info

    # 图训练
    # 不进行 mini-batch
    def user_round_train_graph(self, model, device, n_round,  batch_size, n_round_samples=-1, debug=False):
        X,Y = self.round_data(n_round, n_round_samples)

        # 下面计算当前样本的邻接矩阵
        self.create_adjacency_matrix()

        X, Y = torch.Tensor(X), torch.Tensor(Y.values).long()
        model.train()
        correct = 0
        prediction = []
        real = []
        total_loss = 0
        model, X, Y = model.to(device), X.to(device), Y.to(device)

        output = model(X, self.adj)
        loss = F.nll_loss(output, Y)
        # loss = self.train_loss(output.to(torch.float64), Y) # 修改loss

        total_loss += loss
        loss.backward()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(Y.view_as(pred)).sum().item()
        prediction.extend(pred.reshape(-1).tolist())
        real.extend(Y.reshape(-1).tolist())


        grads = {'n_samples': len(X), 'named_grads': {}}
        for name, param in model.named_parameters():
            grads['named_grads'][name] = param.grad.detach().cpu().numpy()
        
        worker_info = {}
        worker_info["train_acc"] = correct / len(X)
        worker_info["loss"] = total_loss.data

        if debug:
            print('Training Loss: {:<10.2f}, accuracy: {:<8.2f}'.format(
                total_loss, 100. * correct / len(X)))

        return grads, worker_info


    # 下面是借助DGL库实现的一些功能
    def create_dgl_graph(self):
        txId_list = self.remain_txId.values
        for index, txId in enumerate(txId_list):
            self.txId_map[txId] = index
        txId1_list = self.edges['txId1'].values
        txId2_list = self.edges['txId2'].values
        src_list = []
        des_list = []
        for txId1, txId2 in zip(txId1_list, txId2_list):
            if txId1 in self.txId_map.keys() and txId2 in self.txId_map.keys():
                src_list.append(self.txId_map[txId1])
                des_list.append(self.txId_map[txId2])
        num_nodes = len(txId_list)
        # 借助 DGL 建图
        self.g  = dgl.graph((src_list, des_list), num_nodes=num_nodes)
        self.g.ndata['feat'] = torch.Tensor(self.data[0])
        self.g.ndata['class'] = torch.LongTensor(self.data[1])

        # 建立自边
        self.g = dgl.add_self_loop(self.g)
        

    def cal_f1(self, prediction, real):
        # 计算 F1
        tp, tn, fp, fn = 1e-8, 1e-8, 1e-8, 1e-8

        for x, y in zip(prediction, real):
            if x == 1 and y == 1:
                tp += 1
            elif x == 0 and y == 0:
                tn += 1
            elif x == 1 and y == 0:
                fp += 1
            elif x == 0 and y == 1:
                fn += 1

        precision = tn * 1.0 / (tn + fn)
        recall = tn * 1.0 / (tn + fp)
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def dgl_train_graph(self, model, device, n_round,  batch_size, n_round_samples=-1, debug=False):
        
        X = self.g.ndata['feat']
        Y = self.g.ndata['class']
        model.train()
        correct = 0
        prediction = []
        real = []
        total_loss = 0
        model, X, Y = model.to(device), X.to(device), Y.to(device)

        output = model(X)

        # 处理梯度，只计算有标签数据的loss
        loss = 0
        labeled_samples = 0 # 有标签样本的数量
        y_labeled_index = [] # 有标签样本的下标
        for i in range(output.shape[0]):
            if Y[i] != 2: # 有标签的数据
                labeled_samples += 1
                loss += F.nll_loss(output[i].reshape(1, -1), Y[i].reshape(-1))
                # loss += self.cal_loss_need_one_hot(output[i].reshape(1, -1), Y[i].reshape(-1))
                y_labeled_index.append(i)

        loss /= labeled_samples

        # loss = self.cal_loss_need_one_hot(output, Y)

        # loss = F.nll_loss(output, Y)


        total_loss += loss
        loss.backward()

        # 只提取有标签的样本
        output = output[y_labeled_index]
        Y = Y[y_labeled_index]

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(Y.view_as(pred)).sum().item()
        prediction.extend(pred.reshape(-1).tolist())
        real.extend(Y.reshape(-1).tolist())

        # 计算 F1
        f1 = self.cal_f1(prediction, real)


        grads = {'n_samples': len(output), 'named_grads': {}}
        for name, param in model.named_parameters():
            grads['named_grads'][name] = param.grad.detach().cpu().numpy()
        
        worker_info = {}
        worker_info["train_acc"] = correct / len(output)
        worker_info["f1"] = f1
        worker_info["loss"] = total_loss.data

        if debug:
            print('Training Loss: {:<10.2f}, accuracy: {:<8.2f}'.format(
                total_loss, 100. * correct / len(output)))

        return grads, worker_info
        

    

