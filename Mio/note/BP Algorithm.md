# [How the backpropagation algorithm works](http://neuralnetworksanddeeplearning.com/chap2.html)

## 前言

这份笔记主要是记录对BP算法本质的理解，原链接在上方，其中BP3和BP4的证明本来是作业，我直接给出了。

**本笔记需要在typora下浏览。**

**直接阅读原文体验更佳**

## 基本表示

$w_{jk}^{l}$

从第l-1层神经网络的**第k个节点**到第l层神经网络**第j个节点**的权值。

$b_{j}^{l}$

第l层神经网络的第j个神经元的偏差。

$a_{j}^{l}$

第l层神经网络的第j个神经元的激励值（输出值）。

$a^{l}=\sigma(w^{l}a^{l-1}+b^{l})$

二者的关系，注意上述变量均向量化运算。由于之前w是反方向定义的，所以这里w矩阵无需转置。

$z^{l}_{j}$

第l层神经网络第j个神经元的加权输入值。

$z^{l}=w^{l}a^{l-1}+b^{l}$

z和a,b的关系。

n

训练集样本总数。

L

神经网络总层数（从1开始）。

K

输出神经元总数。

## 代价函数

这里采用平方函数。

$C=\frac {1}{2n} \sum_{j=1}^{K}(y_{j}-a_{j}^{L})^{2}$

实际上在这份笔记里C选什么下面都没影响的。

但要注意的是实际上由于我们最终是求对权值的偏导，x和y都是固定的，所以对于**每一组数据**有

$C_{x}=\frac {1}{2}(y-a^{L})^2$

**由于x固定，所以下面推导过程中直接取Cx为C。**

## 点乘

和matlab的.*同理，不多解释了。

## 四个方程

### 反向传播法的目标

首先我们要认识到，我们最终要求的是梯度，然后进行梯度下降，在这个过程中我们使用反向传播法还是为了求**梯度**$\frac {\partial C}{\partial w^{l}_{jk}}$和$\frac {\partial C}{\partial b^{l}_{j}}$。值得说明的是在Ng课程中这二者是统一的。

### 中间量

在反向传播法中，实际并不是直接计算上述两个偏微分，而是先定义了一个中间量。

$\delta_{j}^{l} = \frac {\partial C} {\partial z_{j}^{l}}$

表示第l层神经网络的第j个神经元的**错误值**。

### 错误值的意义

首先考虑第l层神经网络的第j个神经元，如果其输入值$z_{j}^{l}$出现了一点偏差$\Delta z_{j}^{k}$，导致最终代价变化了$\frac {\partial C}{\partial z_{j}^{l}} \Delta z_{j}^{l}$（类比下斜率乘Δx等于Δy）。

这就是$\delta$定义的来源。

### BP1

方程BP1如下

$\delta_{j}^{L}=\frac {\partial C}{\partial a_{j}^{L}} \sigma'(z_{j}^{L})$

向量形式

$\delta^{L}=\nabla_{a}C.*\sigma'(z^{L})$

其中$\nabla_{a}C$是一个由$\frac {\partial C}{\partial a_{j}^{L}}$填满的向量。

#### 意义

首先右边第一项$\frac {\partial C}{\partial a_{j}^{L}}$反映了代价相对最终输出神经元变化的快慢，如果某个输出神经元并不很影响代价，那么相应的这个神经元带来的错误$\delta$也比较小，满足直观感受。

第二项$\sigma'(z_{j}^{L})$反映了激励函数相对于$z_{j}^{L}$的变化速率，同样符合直观感受。

#### 证明

从定义$\delta_{j}^{L} = \frac {\partial C} {\partial z_{j}^{L}}$出发，应用**链式法则**

$\delta_{j}^{L}=\sum_{k=1}^{K}\frac {\partial C}{\partial a_{k}^{L}}\frac {\partial a_{k}^{L}}{\partial z_{j}^{L}}$

显然有，当k!=j的时候$\frac {\partial a_{k}^{L}}{\partial z_{j}^{L}}\equiv0$

（值得一提的是上面这种方式在后面证明的时候会多次用到）

所以可得

$\delta_{j}^{L}=\frac {\partial C}{\partial a_{j}^{L}}\frac {\partial a_{j}^{L}}{\partial z_{j}^{L}}$

接着考虑到$a_{j}^{L}=\sigma(z_{j}^{L})$有$\frac {\partial a_{j}^{L}}{\partial z_{j}^{L}}=\sigma'(z_{j}^{L})$

代入有$\delta_{j}^{L}=\frac {\partial C}{\partial a_{j}^{L}} \sigma'(z_{j}^{L})$

证毕

### BP2

方程BP2如下（已经向量化）

$\delta^{l}=((\omega^{l+1})^{T}\delta^{l+1}).*\sigma'(z^{l})$

#### 意义

本质上就是一个递推式，结合BP1a可以计算出所有的$\delta$了。

#### 证明

依旧是根据定义$\delta_{j}^{l} = \frac {\partial C} {\partial z_{j}^{l}}$应用**链式法则**

（值得注意的是此处是l而不是L，和BP1应用链式法则的情形有差别）
$$
\begin{align}
\delta_{j}^{l}&=\frac {\partial C}{\partial z_{j}^{l}}\\
&=\sum_{k}\frac{\partial C}{\partial z_{k}^{l+1}}\frac {\partial z_{k}^{l+1}} {\partial z_{j}^{l}}\\
&=\sum_{k} \frac {\partial z_{k}^{l+1}} {\partial z_{j}^{l}}\delta_{k}^{l+1}
\end{align}
$$
由于我们想出现递推式，所以对$z^{l+1}_{k}$进行链式法则，同时有

$z_{k}^{l+1}=\sum_{j}w_{kj}^{l+1}a_{j}^{l}+b_{k}^{l+1}=\sum_{j}w_{kj}^{l+1}\sigma(z_{j}^{l})+b_{k}^{l+1}$

$\frac {\partial z_{k}^{l+1}} {\partial z_{j}^{l}} = \frac {\partial z_{k}^{l+1}} {\partial \sigma(z_{j}^{l})} \frac {\partial \sigma(z_{j}^{l})} {\partial z_{j}^{l}} = \omega_{kj}^{l+1}\sigma'(z_{j}^{l})$

（第二个式子其实也应用了链式法则，我没有具体写出，过程和BP1类似，只有j的时候第二项非零）

带回上面的式子可以得到

$\delta_{j}^{l}=\sum_{k}\omega_{kj}^{l+1}\delta_{k}^{l+1}\sigma'(z_{j}^{l})$

这正是BP2的一般表达式。

### BP3

方程BP3如下

$\frac {\partial C}{\partial b_{j}^{l}}=\delta_{j}^{l}$

向量形式

$\frac {\partial C}{\partial b} = \delta$

#### 意义

刚才已经提到了我们可以计算出所有的$\delta$，这样目标就完成一半了。

#### 证明

对$\frac {\partial C}{\partial b_{j}^{i}}$应用链式法则，只有取j的时候非零可得

$\frac {\partial C}{\partial b_{j}^{i}}=\frac {\partial C}{\partial z_{j}^{l}} \frac {\partial z_{j}^{l}}{\partial b_{j}^{l}}$

然而

$z^{l}=w^{l}a^{l-1}+b^{l}$

显然$\frac {\partial z_{j}^{l}}{\partial b_{j}^{l}}\equiv1$

而$\frac {\partial C}{\partial z_{j}^{l}}=\delta_{j}^{l}$

证毕。

### BP4

方程BP4如下

$\frac {\partial C}{\partial \omega_{jk}^{l}}=a_{k}^{l-1}\delta_{j}^{l}$

简化形式

$\frac {\partial C}{\partial \omega}=a_{in}\delta_{out}$

#### 意义

直观来看，当一个神经元输出值较小的时候，可以认为这个权值学习速度较慢。

另外值得一提的是，在Ng的课程中可以认为偏差单元是$a_{k}^{l-1}\equiv1$进而把BP3和BP4联系起来。

#### 证明

依旧是对$\frac {\partial C}{\partial w_{jk}^{l}}$链式法则，只有取j的时候非零得

$\frac {\partial C}{\partial z_{j}^{i}}=\frac {\partial C}{\partial z_{j}^{l}}\frac {\partial z_{j}^{l}}{\partial \omega_{jk}^{l}}=\delta_{j}^{l}a_{k}^{l-1}$

直接证毕。

### 总结

> Summing up, we've learnt that a weight will learn slowly if either the input neuron is low-activation, or if the output neuron has saturated, i.e., is either high- or low-activation.

![](http://neuralnetworksanddeeplearning.com/images/tikz21.png)

## 后记

在学习反向传播法的时候走了许多弯路，同时由于好奇心过于旺盛不满足了解表面而深入研究浪费了不少时间，不过我感觉很有用的。不仅了解了BP算法的核心思想，同时四个方程反映出来的特征也有助于对整个神经网络的理解。

此外吐槽下国内教程真的垃圾，还是直接看外文原版讲义省事。

## 附注

唯一比较有价值的是知乎这个提问

https://www.zhihu.com/question/27239198?rf=24827633

其中有几张图粘过来相信有助理解。

![](https://pic1.zhimg.com/v2-02970ed9be998fd5ca34574a8d9882cc_b.jpg)

![](https://pic2.zhimg.com/v2-14d27603478c887d868fc4702b853d59_b.jpg)

![](https://pic4.zhimg.com/v2-0898d65213144d497d0342a550273957_b.jpg)

![](https://pic3.zhimg.com/v2-d7a5ac9279761efd23bc351bb5d484d6_b.jpg)

![](https://pic4.zhimg.com/v2-00d8a4abd58aad3d7b4f5bf338cee98b_b.jpg)

![](https://pic2.zhimg.com/v2-20aee4359cf4fefd0e44c8560095d7e5_b.jpg)

