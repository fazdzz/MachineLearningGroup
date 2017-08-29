# Week 9

之前Week8是神经网络基本介绍，不留笔记了。

本周内容比较重要。

## 神经网络

### 代价函数

和之前的模型一样，神经网络也有自己的代价函数。

Ng这里类比逻辑回归给出了带正规化的代价函数。

![](http://latex.codecogs.com/gif.latex?J%28%5Ctheta%29%3D-%5Cfrac%7B1%7D%7Bm%7D%5B%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5Csum_%7Bk%3D1%7D%5E%7BK%7Dy_%7Bk%7D%5E%7B%28i%29%7Dlog%28h_%7B%5Ctheta%7D%28x%5E%7B%28i%29%7D%29%29_%7Bk%7D&plus;%281-y_%7Bk%7D%5E%7B%28i%29%7D%29log%281-%28h_%7B%5Ctheta%7D%28x%5E%7B%28i%29%7D%29%29_%7Bk%7D%29%5D%20&plus;%20%5Cfrac%20%7B%5Clambda%7D%7B2m%7D%5Csum_%7Bl%3D1%7D%5E%7BL-1%7D%5Csum_%7Bi%3D1%7D%5E%7Bs_%7Bl%7D%7D%5Csum_%7Bj%3D1%7D%5E%7Bs_%7Bl%7D&plus;1%7D%28%5Ctheta_%7Bji%7D%5E%7B%28l%29%7D%29%5E%7B2%7D)

其实道理还是一致的。

### 梯度下降

由于还是依靠梯度下降，所以要计算梯度。

直接正向计算复杂度很大（路径重复），因此采用了反向传播算法。

详情可以参考[BP Algorithm](./BP Algorithm.md)

一旦这个理解了，可以说基本就掌握神经网络核心了，因为BP算法四个方程也反映了神经网络的一般特征。

### 梯度检查

上述BP算法是非常复杂的，出错可能性也不小，好在Ng为我们提供了一种Debug方法。

![](http://latex.codecogs.com/gif.latex?%5Cfrac%20%7B%5Cpartial%7D%7B%5Cpartial%20%5Ctheta_%7Bi%7D%7DJ%28%5Ctheta%29%20%5Capprox%20%5Cfrac%20%7BJ%28%5Ctheta_%7B1%7D%2C%5Ctheta_%7B2%7D...%5Ctheta_%7Bi%7D&plus;%5Cepsilon...%5Ctheta_%7Bn%7D%29-J%28%5Ctheta_%7B1%7D%2C%5Ctheta_%7B2%7D...%5Ctheta_%7Bi%7D-%5Cepsilon...%5Ctheta_%7Bn%7D%29%7D%7B2%5Cepsilon%7D)

实质上就是利用导数的定义计算偏导的近似值然后判断。

### 随机初始化

Ng还提到了一个问题，就是如果θ初始量全部设置成一样的话，会导致某些权值始终相等，进而导致整个神经网络出现冗余。

解决方法也很简单，就是使用rand()来生成一些初始值。