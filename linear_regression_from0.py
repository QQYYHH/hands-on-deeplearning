import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

# 下面两个函数画图相关
def use_svg_display():
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

# 每次随机读取 训练集里 batch_size大小的数据
# 这个函数是迭代版本，可以不用事先生成全部 batch
# 迭代，一个一个返回，借助yield实现
'''
for X, y in data_iter(...):
    operations with X and y
'''
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个 batch
        # yield 可以立刻返回当前循环内的数据，后续数据下一次迭代获取
        # index_select(dim, pos), 从第dim纬，选pos位置上的参数，pos是Tensor
        yield features.index_select(0, j), labels.index_select(0, j)

# 定义线性回归模型
def linreg(X, w, b):
    return torch.mm(X, w) + b

# 定义损失函数
# 这里用的是方差, 也就是pytorch中的 MSELoss 但这里是未求和之前，各个样本的平方差向量
def squared_loss(y_hat, y):
    # 注意这里返回的是向量
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

# 定义优化算法，小批量随机梯度下降算法 优化每一个权重
def sgd(params, lr, batch_size):
    for param in params:
        # 注意这里用的是 param.data，避免干扰自动计算的梯度
        param.data -= lr * param.grad / batch_size

# 首先构造简单人工数据集
num_inputs = 2 # 特征数量
num_examples = 1000 # 训练样本数
true_w = [2, -3.4] # 真实特征权重
true_b = 4.2 # 真实 bias
features = torch.randn(num_examples, num_inputs,
                       dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
# 为真实结果添加随机 噪声，正态分布，均值 = 0，标准差 = 0.01
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float32)

# 下面初始化模型参数
# 权重借助正态分布，初始化为 num_inputs x 1 的tensor, 偏差初始化为0
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)



# 训练模型
batch_size = 10 # 小批量训练集的大小
lr = 0.03 # 超参数 学习率
num_epochs = 3 # 超参数 迭代次数
net = linreg # 函数指针
loss = squared_loss

for epoch in range(num_epochs):
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）
    # X和y 分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum() # l 是有关小批量 X和y 的损失，是一个标量
        l.backward() # 小批量的损失 对 模型参数求梯度，也就是求偏导，然后再将当前参数的值代入
        sgd([w, b], lr, batch_size) # 使用小批量随机梯度下降迭代 模型参数

        # 不要忘了梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))


# 训练完成后，比较训练出来的参数和 真实参数值
print(true_w, '\n', w)
print(true_b, '\n', b)
