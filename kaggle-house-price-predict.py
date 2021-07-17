import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.utils.data as Data

torch.set_default_tensor_type(torch.FloatTensor)

# 获取数据
# 数据下载至 Datasets/House-Predict-kaggle下
# 每组样本，第一个特征是 房子实际的价格
train_data = pd.read_csv('/Users/milktime/AnacondaProjects/deeplearning/Datasets/House-Predict-kaggle/train.csv')
test_data = pd.read_csv('/Users/milktime/AnacondaProjects/deeplearning/Datasets/House-Predict-kaggle/test.csv')

# 查看 训练数据前4个样本的 前4个特征和后两个特征，最后一列是 实际的价钱
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# 第一列是ID，对我们的预测没有帮助，因此，将第一列特征舍弃
# 训练集 第一列 和 最后一列不要，测试集 第一列不要
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 数据预处理 - 关键
# 对连续数值的特征做 标准化（standardization）：对每个特征 先减去 均值，再除以 标准差，使得数据服从 标准的正态分布
# 对于缺失的特征，替换成特征的均值

# 选取类型为数值的 特征 列索引
numeric_features_index = all_features.dtypes[all_features.dtypes != 'object'].index
# 对所有 类型为数值 的列（向量） 做标准化
all_features[numeric_features_index] = all_features[numeric_features_index].apply(
    lambda x: (x - x.mean()) / (x.std())
)
# 标准化后，每个数值特征的均值 变为0，所以可以直接用0填充 缺失的特征值
all_features[numeric_features_index] = all_features[numeric_features_index].fillna(0)

# 接下来将离散数值转成指示特征。
# 举个例子，假设特征MSZoning里面有两个不同的离散值RL和RM，那么这一步转换将去掉MSZoning特征，并新加两个特征MSZoning_RL和MSZoning_RM，其值为0或1。
# 如果一个样本原来在MSZoning里的值为RL，那么有MSZoning_RL=1且MSZoning_RM=0。
# 可能会增加特征的数量，这一步就将 特征数 从79增加到 331
# dummy_na=True 将缺失值也当作合法的特征 并为其创建指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)

n_train = train_data.shape[0]
# 通过values属性 得到Numpy格式的数据
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(train_data.iloc[:, -1].values, dtype=torch.float).view(-1, 1)




# 初始化模型（参数）
# 线性回归模型 + 平方损失函数
num_input = train_features.shape[1]
num_output = 1
net = nn.Linear(num_input, num_output)
for param in net.parameters():
    nn.init.normal_(param, mean=0, std=0.01)

loss = nn.MSELoss()

# 定义比赛用的模型评估误差
def evaluate_log_rmse(net, features, labels):
    # 评估的过程不用 计算梯度
    with torch.no_grad():
        # 将小于1 的值设置成1，使得取对数时数值更稳定
        pred = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(loss(pred.log(), labels.log()))
    return rmse.item()



# 模型训练
# weight_decay：权重衰减系数，越大，权重衰减越大，避免过拟合
def train(net, train_features, train_labels, test_features, test_lables, \
    num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_set = Data.TensorDataset(train_features, train_labels)
    train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
    # 这里用 Adam优化算法
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        train_ls.append(evaluate_log_rmse(net, train_features, train_labels))
        if test_lables is not None:
            test_ls.append(evaluate_log_rmse(net, test_features, test_lables))
        # print('epoch %d, loss is %lf' % (epoch, train_ls[-1]))
    return train_ls, test_ls



# 使用K折 交叉验证
# k-fold across verify
# 将训练集分为 k 个集合，进行k轮训练，每次选其中一个为测试集，其余k - 1个为 训练集
# 返回第 i 折需要的训练和验证数据
# 也就是 第i 个 fold 作为测试集，其余作为训练集
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k # 整除以 k
    X_train, y_train = None, None
    for j in range(k):
        # 切片 - slice(start, stop, step)
        # 返回切片对象，将切片对象作用于数组，可截取对应区间的数据
        idx = slice(j * fold_size, (j + 1) * fold_size) 
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            # 测试集
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, learning_rate, \
    weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0.0, 0.0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        # *data，表示data这个形参 可能代表多个参数，相当于C里面的不定参数
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]

        print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))

    return train_l_sum / k, valid_l_sum / k

# 训练模型 k - cross valid
k = 5
num_epochs, lr = 100, 5
weight_decay = 0.002
batch_size = 64


train_l_avg, valid_l_avg = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print('%d-fold validation: avg train rmse %lf, avg valid rmse %lf' % (k, train_l_avg, valid_l_avg))

# 有时候你会发现一组参数的 训练误差可以达到很低，但是在K折交叉验证上的误差可能反而较高。
# 这种现象很可能是由过拟合造成的。因此，当训练误差降低时，我们要观察K折交叉验证上的误差是否也相应降低。




# 下面将预测结果以给定的格式呈现
preds = net(test_features).detach().numpy()
test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
# 按 列 拼接
submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
submission.to_csv('submission.csv', index=False)




# 画图直观看一下
import matplotlib.pyplot as plt
from IPython import display
# 下面两个函数画图相关
def use_svg_display():
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.savefig('temp.jpg')

# semilogy([i + 1 for i in range(100)], train_ls, 'epoch', 'loss')
