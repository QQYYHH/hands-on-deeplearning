'''
Author: QQYYHH
Date: 2022-04-16 16:10:35
LastEditTime: 2022-04-17 16:36:05
LastEditors: QQYYHH
Description: 
FilePath: /src/model.py
welcome to my github: https://github.com/QQYYHH
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data


from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

f = '/Volumes/T7/实验数据/加密流量数据集/Train/flow/tot_norm.csv'
f_test = '/Volumes/T7/实验数据/加密流量数据集/Test/flow/tot_norm.csv'

def get_data_set(df: pd.DataFrame):
    x = df.iloc[:, 6: -1].to_numpy()
    x[x == np.inf] = 1.
    x[np.isnan(x)] = 0.
    y = df['type'].to_numpy().astype(int)
    return x, y

def balance(x, y):
    smo = SMOTE(random_state=42)
    return smo.fit_sample(x, y)

df = pd.read_csv(f)
df_test = pd.read_csv(f_test)

x_train, y_train = get_data_set(df)
# x_train, y_train = balance(x_train, y_train)
x_test, y_test = get_data_set(df_test)

# 解决样本不均衡问题设计的损失函数
class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.gamma = 2
        self.alpha = 0.75
        
    def forward(self, y_pred, y_true):
        # important to add reduction='none' to keep per-batch-item loss
        ce_loss = F.cross_entropy(y_pred, y_true, reduction='none')
        pt = torch.exp(-ce_loss)
        # mean over the batch
        fl = (self.alpha * (1-pt)**self.gamma * ce_loss).mean() 
        return fl

        
class MLP(nn.Module):
    def __init__(self, input_sz: list, dropout_prob=0.2):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(input_sz)):
            self.layers.append(nn.Linear(input_sz[i - 1], input_sz[i]))
        self.drop_out = nn.Dropout(p=dropout_prob)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(self.drop_out(x))
            if i < len(self.layers) - 1:
                # 如果不是最后一层，激活
                x = F.relu(x)
        return x


def paint(x, y):
    data = dict.fromkeys([0, 1, 2, 3, 4])
    for k in data.keys():
        data[k] = [False for i in range(len(y))]

    for i in range(len(y)):
        yy = y[i]
        data[yy][i] = True
    tsne = TSNE(n_components=2)
    tsne.fit_transform(x)
    x = tsne.embedding_
    print('tsne down...')
    color = ['r', 'b', 'c', 'g', 'm']
    lds = []
    for i in range(5):
        lds.append(plt.scatter(x[data[i], 0], x[data[i], 1], c=color[i]))

    plt.legend(lds, ['chat', 'email', 'audio', 'streaming', 'ft'], loc='best')
    plt.savefig('cluster_test.png')
    plt.show()
# paint(x, y)

def rf():
    rf3 = RandomForestClassifier(max_depth=13, min_samples_split=2, min_samples_leaf=10, oob_score=True, random_state=10)
    model = rf3.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("===============================================")
    print(classification_report(y_test, y_pred))

def gbdt():
    param = {
        "num_leaves": 75, 
        "num_trees": 100, 
        "objective": "multiclass", 
        "num_class": 5, 
        "num_iterations": 100, 
        "is_unbalance": True, 
        "verbose": -1,  
    }

    train_data = lgb.Dataset(data=x_train,label=y_train)
    valid_data = lgb.Dataset(data=x_test,label=y_test)

    clf = lgb.train(param, train_data, valid_sets=[valid_data])
    y_pred = clf.predict(x_test, num_iteration=clf.best_iteration)
    y_pred = y_pred.argmax(axis=1)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("===============================================")
    print(classification_report(y_test, y_pred))

def mlp():
    global x_train, y_train, x_test, y_test
    x_train, y_train = torch.from_numpy(x_train.astype(np.float32)), torch.from_numpy(y_train.astype(np.longlong))
    x_test, y_test = torch.from_numpy(x_test.astype(np.float32)), torch.from_numpy(y_test.astype(np.longlong))

    num_epochs, lr = 100, 0.02
    weight_decay = 0.002
    batch_size = 64
    net = MLP([76, 32, 5])
    loss = FocalLoss()

    train_set = Data.TensorDataset(x_train, y_train)
    train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
    # weight_decay 为正则项
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=weight_decay)
    # train
    for epoch in range(num_epochs):
        for x, y in train_iter:
            l = loss(net(x), y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        print('[epoch %d], train loss is %lf' % (epoch, l))

    y_pred = net(x_test)
    y_pred = y_pred.argmax(axis=1)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("===============================================")
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    gbdt()
