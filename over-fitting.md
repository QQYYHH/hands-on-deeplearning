# overfitting 的解决办法
## 1 what is over-fitting
当训练模型过于复杂时，或者模型的参数数量 > 样本数，模型就会出现over-fitting现象
**表现** 模型在训练集上的准确率 远大于 测试集

## 2 应对方法
### 2.1 权重衰减
> https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter03_DL-basics/3.12_weight-decay 

权重衰减等价于 L2范数正则化（regularization），在原始损失函数的基础上 添加 L2范数惩罚项，即模型参数的平方和 
假设 loss = 
<a href="https://www.codecogs.com/eqnedit.php?latex=loss&space;=&space;l(w_1,&space;w_2,&space;b)&space;=&space;\frac{1}{n}&space;\sum^n_{i&space;=&space;1}&space;(w_1x_1^{(i)}&space;&plus;&space;w_2x_2^{(i)}&space;&plus;&space;b&space;-&space;y^{(i)})^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?loss&space;=&space;l(w_1,&space;w_2,&space;b)&space;=&space;\frac{1}{n}&space;\sum^n_{i&space;=&space;1}&space;(w_1x_1^{(i)}&space;&plus;&space;w_2x_2^{(i)}&space;&plus;&space;b&space;-&space;y^{(i)})^2" title="loss = l(w_1, w_2, b) = \frac{1}{n} \sum^n_{i = 1} (w_1x_1^{(i)} + w_2x_2^{(i)} + b - y^{(i)})^2" /></a>
<br>
那么经过权重衰减后的 损失函数 为：<br>
<a href="https://www.codecogs.com/eqnedit.php?latex=l(w_1,&space;w_2,&space;b)&space;&plus;&space;\frac{\lambda}{2n}||w||^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l(w_1,&space;w_2,&space;b)&space;&plus;&space;\frac{\lambda}{2n}||w||^2" title="l(w_1, w_2, b) + \frac{\lambda}{2n}||w||^2" /></a>
<br>
<a href="https://www.codecogs.com/eqnedit.php?latex=||w||^2&space;=&space;w_1^2&space;&plus;&space;w_2^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?||w||^2&space;=&space;w_1^2&space;&plus;&space;w_2^2" title="||w||^2 = w_1^2 + w_2^2" /></a>
<br>
当 权重衰减的超参数 lambda 过大时，权值参数的各个 元素就会尽可能小，在一定程度上 削减了 over-fitting程度

### 2.2 丢弃法
随机丢弃 某网络层输出 的单元，每层的网络单元有p的概率被丢弃，有1 - p的概率 除以 1 - p <br>
设超参数 <a href="https://www.codecogs.com/eqnedit.php?latex=\xi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\xi" title="\xi" /></a>为 0 和 1 的概率分别为 p 和 1 - p <br>
<a href="https://www.codecogs.com/eqnedit.php?latex=h_i^,&space;=&space;\frac{\xi_i}{1-p}h_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_i^,&space;=&space;\frac{\xi_i}{1-p}h_i" title="h_i^, = \frac{\xi_i}{1-p}h_i" /></a> <br>
由于 <a href="https://www.codecogs.com/eqnedit.php?latex=E(\xi_i)&space;=&space;1&space;-&space;p" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E(\xi_i)&space;=&space;1&space;-&space;p" title="E(\xi_i) = 1 - p" /></a> ，因此 <br>
<a href="https://www.codecogs.com/eqnedit.php?latex=E(h_i^),&space;=&space;\frac{E(\xi_i)}{1-p}&space;h_i&space;=&space;h_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E(h_i^),&space;=&space;\frac{E(\xi_i)}{1-p}&space;h_i&space;=&space;h_i" title="E(h_i^), = \frac{E(\xi_i)}{1-p} h_i = h_i" /></a> <br>
即 丢弃法 不改变其输入的期望值
