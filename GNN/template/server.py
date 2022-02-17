from datetime import datetime, time
import os
import sys
import shutil
import unittest
import pickle
import pandas as pd


import numpy as np
from numpy.lib import real
from sklearn.metrics import classification_report
import torch
import torch.nn.functional as F
from context import FederatedSGD
from context import PytorchModel
from learning_model import *
from preprocess import get_test_data



class ParameterServer(object):
    def __init__(self, init_model_path, testworkdir, resultdir):
        self.round = 0
        self.rounds_info = {}
        self.rounds_model_path = {}
        self.worker_info = {}
        self.current_round_grads = []
        self.init_model_path = init_model_path
        self.aggr = FederatedSGD(
            model=PytorchModel(torch=torch,
                               model_class=BTCModel,
                               init_model_path=self.init_model_path,
                               optim_name='Adam'),
            framework='pytorch',
        )
        self.testworkdir = testworkdir
        self.RESULT_DIR = resultdir
        if not os.path.exists(self.testworkdir):
            os.makedirs(self.testworkdir)
        if not os.path.exists(self.RESULT_DIR):
            os.makedirs(self.RESULT_DIR)
   
        # test_edges columns - 'txId1' 'txId2' 'Timestep'
        self.test_data,self.test_edges = get_test_data()

        self.round_train_acc= []
        self.round_train_loss = []
        self.round_train_f1 = [] # 增加F1计算
        self.adj = None # 根据测试集构建的临接矩阵
        self.remain_txId = []
        self.txId_map = {}

        # DGL相关
        # 保存 9组图数据
        self.graphs = []
        
        # 9组图数据的 txId映射关系
        # timestep -> {txId: index}
        self.txId_maps9 = {}

        # 9组图数据中的 交易数量
        # timestep -> cnt
        self.maps9_cnt = {}

        # timestep -> [index1, index2, ...]
        # 不同测试图中 节点在完整数据集中的 index，便于后续从原始数据中 提取不同测试图的特征
        # 也便于恢复数据
        self.tot_index = {} 

        # 先注释掉，便于调试
        self.preprocess_test_data()

  

    def preprocess_test_data(self):

        self.predict_data = self.test_data[self.test_data['class']==3] # to be predicted

        self.predict_data_txId = self.predict_data[['txId','Timestep']].reset_index(drop=True)

        self.remain_txId = self.predict_data_txId['txId']
        self.remain_txId = self.remain_txId.reset_index(drop=True)

        # x = self.predict_data.iloc[:, 3:]
        feature_name =['Local_feature_61', 'Local_feature_25', 'Aggregate_feature_47', 'Aggregate_feature_11', 'Aggregate_feature_62', 'Local_feature_58', 'Aggregate_feature_16', 'Aggregate_feature_24', 'Aggregate_feature_69', 'Aggregate_feature_17', 'Aggregate_feature_43', 'Local_feature_87', 'Aggregate_feature_23', 'Aggregate_feature_64', 'Aggregate_feature_31', 'Local_feature_32', 'Local_feature_46', 'Local_feature_44', 'Local_feature_80', 'Aggregate_feature_18', 'Local_feature_57', 'Local_feature_91', 'Local_feature_51', 'Local_feature_52', 'Aggregate_feature_56', 'Aggregate_feature_9', 'Aggregate_feature_63', 'Local_feature_34', 'Aggregate_feature_10', 'Aggregate_feature_40', 'Local_feature_30', 'Local_feature_11', 'Local_feature_37', 'Aggregate_feature_38', 'Local_feature_4', 'Aggregate_feature_66', 'Local_feature_16', 'Local_feature_28', 'Local_feature_60', 'Aggregate_feature_12', 'Local_feature_9', 'Aggregate_feature_42', 'Local_feature_35', 'Aggregate_feature_32', 'Local_feature_8', 'Aggregate_feature_8', 'Aggregate_feature_55', 'Aggregate_feature_29', 'Local_feature_55', 'Local_feature_83', 'Local_feature_45', 'Aggregate_feature_1', 'Aggregate_feature_14', 'Local_feature_43', 'Local_feature_65', 'Local_feature_73', 'Aggregate_feature_71', 'Local_feature_56', 'Aggregate_feature_20', 'Local_feature_1', 'Local_feature_66', 'Local_feature_63', 'Aggregate_feature_51', 'Local_feature_19', 'Local_feature_50', 'Aggregate_feature_35', 'Aggregate_feature_46', 'Local_feature_21', 'Local_feature_53', 'Local_feature_20', 'Aggregate_feature_41', 'Aggregate_feature_19', 'Local_feature_79', 'Local_feature_86', 'Local_feature_7', 'Aggregate_feature_44', 'Aggregate_feature_27', 'Local_feature_17', 'Local_feature_42', 'Aggregate_feature_67', 'Local_feature_93', 'Local_feature_10', 'Local_feature_36', 'Aggregate_feature_53', 'Aggregate_feature_34', 'Local_feature_69', 'Aggregate_feature_28', 'Aggregate_feature_48', 'Aggregate_feature_61', 'Local_feature_3', 'Aggregate_feature_52', 'Aggregate_feature_30', 'Aggregate_feature_15', 'Aggregate_feature_57', 'Local_feature_89', 'Local_feature_48', 'Local_feature_12', 'Local_feature_78', 'Local_feature_47', 'Local_feature_26', 'Aggregate_feature_49', 'Aggregate_feature_45', 'Local_feature_14', 'Local_feature_33', 'Aggregate_feature_3', 'Aggregate_feature_4', 'Local_feature_90', 'Aggregate_feature_54', 'Local_feature_23', 'Aggregate_feature_13', 'Local_feature_64', 'Aggregate_feature_58', 'Local_feature_72', 'Aggregate_feature_21', 'Aggregate_feature_6', 'Aggregate_feature_22', 'Local_feature_15', 'Aggregate_feature_68', 'Local_feature_13', 'Aggregate_feature_72', 'Local_feature_75', 'Aggregate_feature_25', 'Local_feature_18', 'Aggregate_feature_5', 'Aggregate_feature_50', 'Aggregate_feature_60', 'Aggregate_feature_36', 'Local_feature_22', 'Aggregate_feature_70', 'Local_feature_24', 'Local_feature_27', 'Local_feature_82', 'Local_feature_39', 'Local_feature_62', 'Local_feature_5', 'Aggregate_feature_33', 'Local_feature_29', 'Local_feature_38', 'Aggregate_feature_37', 'Local_feature_70', 'Local_feature_92', 'Aggregate_feature_7', 'Aggregate_feature_39', 'Local_feature_85', 'Aggregate_feature_2', 'Local_feature_81', 'Local_feature_88', 'Local_feature_49', 'Local_feature_74', 'Local_feature_54']
        x = self.predict_data.loc[:, feature_name]

        x = x.reset_index(drop=True)
        x = x.to_numpy().astype(np.float32)
        x[x == np.inf] = 1.
        x[np.isnan(x)] = 0.

        # mean = pd.DataFrame(x).mean().tolist()
        # std = pd.DataFrame(x).std().tolist()
        # for j in range(0,len(mean)):
        #     for i in range(0,len(x)):
        #         x[i][j] = (x[i][j]-mean[j])/std[j]



        self.predict_data = x
        # 根据测试集构建临接矩阵
        # self.create_adjacency_matrix()
        # 建立9个测试图
        self.create_test_graph()
        for index, g in enumerate(self.graphs):
            print('timestep: ', index, ':', 'number of nodes: ', g.num_nodes(), 'number of edges: ', g.num_edges())


    # aggregate mean and std from workers
    def aggregate_mean_and_std(self,x_mean_all_worker, x_std_all_worker, n_user, n_sample_worker):
        agg_mean_all = []
        agg_std_all = []

        # 结合测试集的数据分布
        for g in self.graphs:
            x = g.ndata['feat']
            x_mean = pd.DataFrame(x).mean().tolist()
            x_std = pd.DataFrame(x).std().tolist()
            n_sample_test = len(x)
            x_mean_all_worker.append(x_mean)
            x_std_all_worker.append(x_std)
            n_sample_worker.append(n_sample_test)

        # 每一个特征都聚合
        for j in range(0,len(x_mean_all_worker[0])):
            cur_mean = x_mean_all_worker[0][j]
            cur_sample_sum = n_sample_worker[0]
            cur_std = x_std_all_worker[0][j]
            for i in range(0,n_user):
                n1, n2, sd1, sd2 = cur_sample_sum, n_sample_worker[i], cur_std, x_std_all_worker[i][j]
                m1, m2 = cur_mean, x_mean_all_worker[i][j]
                # 计算这一轮均值
                res_mean = (n1 * m1 + n2 * m2) / (n1 + n2)
                # 计算这一轮std
                up = (n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2 + (n1 * n2) / (n1 + n2) * (m1 ** 2 + m2 ** 2 - 2 * m1 * m2)
                down = n1 + n2 - 1
                res_std = np.sqrt(up / down)

                cur_mean, cur_std = res_mean, res_std

            agg_mean_all.append(cur_mean)
            agg_std_all.append(cur_std)

        return agg_mean_all, agg_std_all


    def feature_standard(self,agg_mean_all,agg_std_all):
        for j in range(0,len(agg_mean_all)):
            for i in range(0,self.predict_data.shape[1]):
                self.predict_data[i][j] = (self.predict_data[i][j]-agg_mean_all[j])/agg_std_all[j]

        # 新增建图函数 - 构建邻接矩阵
    def create_adjacency_matrix(self):
        # 对txId 离散化，映射为连续的 编号，便于求解邻接矩阵
        txId_list = self.remain_txId.values

        for index, txId in enumerate(txId_list):
            self.txId_map[txId] = index
        
        # 下面进行具体建图操作
        N = len(txId_list)
        print('size of test-adjacency is, ', N)
        self.adj = torch.zeros(N, N)
        txId1_list = self.test_edges['txId1'].values
        txId2_list = self.test_edges['txId2'].values
        for txId1, txId2 in zip(txId1_list, txId2_list):
            if txId1 in self.txId_map.keys() and txId2 in self.txId_map.keys():
                self.adj[self.txId_map[txId1], self.txId_map[txId2]] = 1


    def get_latest_model(self):
        if not self.rounds_model_path:
            return self.init_model_path

        if self.round in self.rounds_model_path:
            return self.rounds_model_path[self.round]

        return self.rounds_model_path[self.round - 1]

    def receive_grads_info(self, grads): # receive grads info from worker
        self.current_round_grads.append(grads)
    
    def receive_worker_info(self, info): # receive worker info from worker
        self.worker_info = info
    
    def process_round_train_acc(self): # process the "round_train_acc" info from worker
        self.round_train_acc.append(self.worker_info["train_acc"])
        # 也添加loss 信息
        self.round_train_loss.append(self.worker_info["loss"])
        # 添加F1
        self.round_train_f1.append(self.worker_info["f1"])

    
    def print_round_train_acc(self):
        mean_round_train_acc = np.mean(self.round_train_acc) * 100
        print("\nMean_round_train_acc: ", "%.2f%%" % (mean_round_train_acc))

        mean_round_train_loss = np.mean(self.round_train_loss)
        print("\nMean_round_train_loss: ", "%.2f" % (mean_round_train_loss))

        mena_round_train_f1 = np.mean(self.round_train_f1) * 100
        print("\nMean_round_train_f1: ", "%.2f" % (mena_round_train_f1))
        self.round_train_acc = []
        self.round_train_loss = []
        self.round_train_f1 = []
        return {"mean_round_train_acc":mean_round_train_acc
               }

    def aggregate(self):
        self.aggr(self.current_round_grads)

        path = os.path.join(self.testworkdir,
                            'round-{round}-model.md'.format(round=self.round))
        self.rounds_model_path[self.round] = path
        if (self.round - 1) in self.rounds_model_path:
            if os.path.exists(self.rounds_model_path[self.round - 1]):
                os.remove(self.rounds_model_path[self.round - 1])

        info = self.aggr.save_model(path=path)

        self.round += 1
        self.current_round_grads = []

        return info
    
    def save_prediction(self, predition):

        predition.to_csv(os.path.join(self.RESULT_DIR, 'result.csv'),index=0)
    
    def save_model(self,model):

        with open(os.path.join(self.RESULT_DIR, 'model.pkl'), 'wb') as fout:
            pickle.dump(model,fout)

    def save_testdata_prediction(self, model, device, test_batch_size):
        # self.dgl_test_train(model, device, test_batch_size)
        # return 
        self.test_data
        loader = torch.utils.data.DataLoader(
            self.predict_data,
            batch_size=test_batch_size,
            shuffle=False,
        )
        prediction = []
        # with torch.no_grad():
        #     for data in loader:
        #         pred = model(data.to(device)).argmax(dim=1, keepdim=True)
        #         prediction.extend(pred.reshape(-1).tolist())
        # self.predict_data_txId['prediction'] = prediction


        # 注意后面删去这些
        from preprocess import realclass
        f1_list = []
        prediction = [x for x in range(len(self.predict_data))]
        with torch.no_grad():
            for timestep, g in enumerate(self.graphs):
                    X = g.ndata['feat']
                    pred = model(X).argmax(dim=1, keepdim=True).reshape(-1).tolist()
                    pre_list = []
                    real_list = []
                    for i, val in enumerate(pred):
                        prediction[self.tot_index[timestep][i]] = val
                        pre_list.append(val)
                        real_list.append(realclass[self.tot_index[timestep][i]])
                    f1_list.append(self.cal_f1(pre_list, real_list))
                    
        self.predict_data_txId['prediction'] = prediction

        f1 = np.mean(f1_list)
        print('test f1 is : {:<.2f}'.format(100. * f1))
        
        self.save_prediction(self.predict_data_txId)
        self.save_model(model)

    # 直接跑完，不进行mini-batch划分
    def my_testdata_prediction(self, model, device, test_batch_size):
        prediction = []
        X = torch.Tensor(self.predict_data)
        with torch.no_grad():
            pred = model(X.to(device), self.adj).argmax(dim=1, keepdim=True)
            prediction.extend(pred.reshape(-1).tolist())

        self.predict_data_txId['prediction'] = prediction

        self.save_prediction(self.predict_data_txId)
        self.save_model(model)


    # 下面是借助 DGL实现的功能
    # 创建测试集的图，一共有9个，分别使用DGL建图，然后进行节点分类
    def create_test_graph(self):
        # 对每个图数据，先做节点映射
        txId_list = self.predict_data_txId['txId'].values
        timestep_list = self.predict_data_txId['Timestep'].values



        for index, (txId, timestep) in enumerate(zip(txId_list, timestep_list)):
            if timestep not in self.txId_maps9.keys():
                self.txId_maps9[timestep] = {}
                # 初始化交易数量为 0
                self.maps9_cnt[timestep] = 0
                # 初始化 totindex集合
                self.tot_index[timestep] = []

            # txId -> current_index
            self.txId_maps9[timestep][txId] = (self.maps9_cnt[timestep])
            self.maps9_cnt[timestep] += 1
            self.tot_index[timestep].append(index)

        # timestep -> [src_nodes_index]
        src_list = {}
        # timestep -> [des_nodes_index]
        des_list = {}
        txId1_list, txId2_list, timestep_list = self.test_edges['txId1'].values, self.test_edges['txId2'].values, self.test_edges['Timestep'].values
        for index, (txId1, txId2) in enumerate(zip(txId1_list, txId2_list)):

            timestep = timestep_list[index]
            if timestep not in src_list.keys():
                src_list[timestep] = []
                des_list[timestep] = []

            if txId1 in self.txId_maps9[timestep].keys() and txId2 in self.txId_maps9[timestep].keys():
                src_list[timestep].append(self.txId_maps9[timestep][txId1])
                des_list[timestep].append(self.txId_maps9[timestep][txId2])
        
        # 逐一生成 DGL graph, 一共9个
        for timestep in range(9):
            num_nodes = self.maps9_cnt[timestep]
            g = dgl.graph((src_list[timestep], des_list[timestep]), num_nodes=num_nodes)
            # 获取节点特征
            choices = self.tot_index[timestep]
            g.ndata['feat'] = torch.Tensor(self.predict_data[choices])
            # 每个点加上 自己到自己的边
            g = dgl.add_self_loop(g)
            self.graphs.append(g)

    # 计算F1
    def cal_f1(self, pre_list, real_list):
        tp, tn, fp, fn = 1e-8, 1e-8, 1e-8, 1e-8

        for x, y in zip(pre_list, real_list):
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
    
    # 训练
    # 先整图训练
    def dgl_test_train(self, model, device, test_batch_size):
        prediction = [x for x in range(len(self.predict_data))]

        # 这是自己加的，提交前记得删除
        # 删除 f1_list, pre_list, real_list, f1
        from preprocess import realclass
        f1_list = []

        with torch.no_grad():
            for timestep, g in enumerate(self.graphs):
                X = g.ndata['feat']
                pred = model(g, X).argmax(dim=1, keepdim=True).reshape(-1).tolist()
                
                pre_list = []
                real_list = []

                for i, val in enumerate(pred):
                    prediction[self.tot_index[timestep][i]] = val

                    pre_list.append(val)
                    real_list.append(realclass[self.tot_index[timestep][i]])

                f1_list.append(self.cal_f1(pre_list, real_list))
                    

        self.predict_data_txId['prediction'] = prediction



        
        f1 = np.mean(f1_list)
        print('test f1 is : {:<.2f}'.format(100. * f1))

        self.save_prediction(self.predict_data_txId)
        self.save_model(model)

