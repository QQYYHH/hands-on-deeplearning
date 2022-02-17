from datetime import datetime
import os
import sys
import shutil
import unittest

import numpy as np
from sklearn.metrics import classification_report
import torch
import torch.nn.functional as F

from context import FederatedSGD
from context import PytorchModel
from learning_model import *
from worker import Worker
from server import ParameterServer


class FedSGDTestSuit(unittest.TestCase):
    RESULT_DIR = 'result'
    N_VALIDATION = 10000
    TEST_BASE_DIR = '/tmp/'

    def setUp(self):
        self.seed = 0
        self.use_cuda = False
        self.batch_size = 64
        self.test_batch_size = 1000
        self.lr = 0.001
        self.n_max_rounds = 200 # 300是比较合适的
        self.log_interval = 20
        self.n_round_samples = 1600 # 每轮选择的训练集样本数量
        #self.n_round_samples = -1
        self.testbase = self.TEST_BASE_DIR
        # 暂时 设定为 31，后9个作为测试集
        # self.n_users = 40 
        self.n_users = 31
        self.testworkdir = os.path.join(self.testbase, 'competetion-test')

        if not os.path.exists(self.testworkdir):
            os.makedirs(self.testworkdir)

        # /tmp/competetion-test/init_model.md
        self.init_model_path = os.path.join(self.testworkdir, 'init_model.md')
        torch.manual_seed(self.seed)

        #if not os.path.exists(self.init_model_path):
        torch.save(BTCModel().state_dict(), self.init_model_path)

        if not os.path.exists(self.RESULT_DIR):
            os.makedirs(self.RESULT_DIR)

        # 处理 测试集
        self.ps = ParameterServer(init_model_path=self.init_model_path,
                                  testworkdir=self.testworkdir,resultdir=self.RESULT_DIR)
        
        self.workers = []
        for u in range(0, self.n_users):
            self.workers.append(Worker(user_idx=u))

    def _clear(self):
        shutil.rmtree(self.testworkdir)

    def tearDown(self):
        self._clear()

    # 类执行的入口函数
    def test_federated_SGD(self):
        torch.manual_seed(self.seed)
        device = torch.device("cuda" if self.use_cuda else "cpu")
        
        # 训练数据预处理
        x_mean_all_worker = []
        x_std_all_worker = []
        all_n_sample_worker = []
        for u in range(0, self.n_users):
            # 预处理数据
            self.workers[u].preprocess_worker_data()
            
            # 传输均值、方差
            x_mean,x_std,n_sample_worker = self.workers[u].compute_mean_and_standard()
            x_mean_all_worker.append(x_mean)
            x_std_all_worker.append(x_std)
            all_n_sample_worker.append(n_sample_worker)
        #把mean，std给server
        agg_mean_all,agg_std_all = self.ps.aggregate_mean_and_std(x_mean_all_worker, x_std_all_worker, self.n_users, all_n_sample_worker)
        #server把聚合后的mean，std给所有的worker，worker给自己的数据标准化后放到self.data里
        for u in range(0,self.n_users):
            self.workers[u].feature_standard(agg_mean_all,agg_std_all)
        
        # 然后标准化server上的测试集特征
        self.ps.feature_standard(agg_mean_all, agg_std_all)
        
        training_start = datetime.now()
        model = None
        # 设置最大迭代次数 相当于 epoch
        for r in range(1, self.n_max_rounds + 1):

            # 获取模型的保存路径，便于读取
            path = self.ps.get_latest_model()
            start = datetime.now()

            # 迭代训练某个节点上的数据
            for u in range(0, self.n_users):
                # 新增模型
                model = BTCModel()

                #model = FLModel()
                model.load_state_dict(torch.load(path))

                # copy到GPU
                model = model.to(device)
                # 具体训练过程，但是没有进行梯度下降，只是累加梯度
                grads,worker_info = self.workers[u].user_round_train(model=model, device=device, n_round=r, batch_size=self.batch_size, 
                    n_round_samples=self.n_round_samples, debug=False)
                
                self.ps.receive_grads_info(grads=grads)
                self.ps.receive_worker_info(worker_info)       # The transfer of information from the worker to the server requires a call to the "ps.receive_worker_info"

                # 将训练集的准确率 记录在 server里
                self.ps.process_round_train_acc()
   
            # 不同节点上的梯度聚集一下，聚集的方法就是加权求平均，权重就是每个节点这一轮训练的样本数
            # 然后更新模型参数
            self.ps.aggregate()
            print('\nRound {} cost: {}, total training cost: {}'.format(
                r,
                datetime.now() - start,
                datetime.now() - training_start,
            ))
            if  model is not None and r % self.log_interval ==0:
                server_info = self.ps.print_round_train_acc() # print average train acc and return
                for u in range(0, self.n_users):              # transport average train acc to each worker
                    self.workers[u].receive_server_info(server_info) # The transfer of information from the server to the worker requires a call to the "worker.receive_server_info"
                    self.workers[u].process_mean_round_train_acc() # workers do processing
                
                # 使用每轮训练出的模型跑 测试集，并保存
                # 交之前记得 save!!!
                self.ps.save_testdata_prediction(model=model, device=device, test_batch_size=self.test_batch_size)

        # 最后，使用最终训练出来的模型 跑测试集合，并保存
        if model is not None:
            # 交之前记得 save!!!
            self.ps.save_testdata_prediction(model=model, device=device, test_batch_size=self.test_batch_size)
            pass
            


def suite():
    suite = unittest.TestSuite()
    suite.addTest(FedSGDTestSuit('test_federated_SGD'))
    return suite


def main():
    runner = unittest.TextTestRunner()
    runner.run(suite())


if __name__ == '__main__':
    main()
