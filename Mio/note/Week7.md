# Week 7

## 欠拟合和过拟合

字面意思。

欠拟合就是对训练集数据拟合度较差，过拟合就是对训练集数据几乎完全拟合但是无法用来预测。

## 减少过拟合

### 减少特征

- 只选择关联性大的特征。
- 选择其他更合适的模型。

### 正规化

减少特征固然可行，但这样也就减少了我们获得的信息，所以本周讲述的重点就是这个“正规化”。

#### 梯度下降正规化

以逻辑回归为例。

##### 代价函数

在正规化中，代价函数J(θ)发生了变化，在原有基础上添加了惩罚，这也是核心所在。

由于惩罚的存在，可以有效减少过度的拟合。

![](http://latex.codecogs.com/gif.latex?J%28%5Ctheta%29%3D-%5Cfrac%20%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5By%5E%7B%28i%29%7Dlog%28h_%7B%5Ctheta%7D%28x%5E%7B%28i%29%7D%29%29&plus;%281-y%5E%7B%28i%29%7D%29log%281-h_%7B%5Ctheta%7D%28x%5E%7B%28i%29%7D%29%29%5D&plus;%5Cfrac%20%7B%5Clambda%7D%20%7B2m%7D%5Csum_%7Bj%3D1%7D%5E%7Bn%7D%5Ctheta_%7Bj%7D%5E%7B2%7D)

这里值得注意的是后面对θ的累加是从1到n的，也就是说**并不包括θ0。**

##### 梯度下降

由于上面提到的θ0并不参与，所以在梯度下降的时候要**分θ0和θ1~n两种情况**计算梯度，此处依旧是对J求偏导，公式略。

#### 正规方程正规化

Ng依旧没有给出证明，直接给出了公式

![](http://latex.codecogs.com/gif.latex?%5Ctheta%20%3D%20%28X%5E%7BT%7DX&plus;%5Clambda%20%5Cbegin%7Bbmatrix%7D%200%26%20%26%20%26%20%26%20%5C%5C%20%26%201%26%20%26%20%26%20%5C%5C%20%26%20%26%20.%26%20%26%20%5C%5C%20%26%20%26%20%26%20.%26%20%5C%5C%20%26%20%26%20%26%20%261%20%5Cend%7Bbmatrix%7D%29%5E%7B-1%7DX%5E%7BT%7Dy)

另外值得说明一点的是，这个式子中括号里面的矩阵可以证明**一定是可逆的**。