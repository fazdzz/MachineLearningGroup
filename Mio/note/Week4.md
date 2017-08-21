# Week4

（之前Week3就是线代基础，扫了一遍没什么可写的，不写了）

这一周讲了两种回归模型学习算法，梯度下降和正规方程法，第一种之前已经提到过了但不深入。

## 梯度下降

这周提到的是对于n个特征的梯度下降怎么做，由于之前已经有单个特征的梯度下降做铺垫，这里就简化一些。

### 数据预处理(Feature Scaling)

由于机器浮点运算精度是有限的，如果数据的范围差的太远，那么会导致J的曲线非常陡峭，这样在梯度下降的过程中会导致出现非常剧烈的抖动，对结果产生影响，所以需要对数据进行Feature Scaling

这里有一个通用公式

```matlab
x = (x - mean(x))/std(x);
% x -> m维列向量
% mean(x) -> 平均数
% std(x) -> 标准差
```

其中标准差也可以换成极差。

此外需要注意一点的是，一定要把每组特征的平均数和标准差保存下来，这样才能对新数据进行变换和预测。

### 假设函数

这里公式稍有变化

![](http://latex.codecogs.com/gif.latex?h_{\theta}(x)&space;=&space;\theta_{0}&space;&plus;&space;\theta_{1}*x_{1}&plus;\theta_{2}*x&space;&plus;&space;...&space;&plus;\theta_{n}*x_{n})

同时我们也可以对之前的一个问题作更加深入的回答

#### 为什么假设是线性的

对于任一特征，如果我们**希望**它在假设中出现的形式为f(x)比如x^2，那么我们只要用换元法令t=f(x)作一个代换，把f(x)整体作为一个新的特征即可，这样最终的假设还是一个线性的。

比如你可能认为房价应该是一个二次类型，希望假设是

![](http://latex.codecogs.com/gif.latex?h_{\theta}(x)&space;=&space;\theta_{0}&space;&plus;&space;\theta_{1}*x_{1}&plus;\theta_{2}*x^{2}_{1})

这时候我们本来只有一个特征x1却出现了x1和x1^2，这时候我们就可以再加一个特征x2=x1^2，那么假设就还是线性的，这里x2=x1^2就是上面所提到的换元。

（感觉后面的神经网络和这个思想有点类似呐）

### 代价函数

之前已经提到了，这里就列一下式子

![](http://latex.codecogs.com/gif.latex?J_%7B%5Ctheta%7D%28%5Ctheta_%7B1%7D%2C%5Ctheta_%7B2%7D...%5Ctheta_%7Bn%7D%29%3D%5Cfrac%20%7B1%7D%7B2m%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%28h_%5Ctheta%28x%29%5E%7B%28i%29%7D-y%5E%7B%28i%29%7D%29%5E%7B2%7D)

### 梯度下降

直接写出最终计算式

![](http://latex.codecogs.com/gif.latex?%5Ctheta_%7Bj%7D%20%3D%20%5Ctheta_%7Bj%7D%20-%20%5Calpha%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%28h_%7B%5Ctheta%7D%28x%5E%7B%28i%29%7D%29-y%5E%7B%28i%29%7D%29x%5E%7B%28i%29%7D_%7Bj%7D)

这个一定要理清i和j的关系。

## 正规方程

除了上述梯度下降方法以外，还有一种正规方程方法可以用来求θ，那就是正规方程法。

证明在[CS229的讲义](http://cs229.stanford.edu/notes/cs229-notes1.pdf)里面，我不想再写了。

此外可以参考 [链接1](http://www.cnblogs.com/daisyliar/p/7202601.html) [链接2](http://www.cnblogs.com/rcfeng/p/3961800.html)

### 和梯度下降法的比较

#### 优点

- 不用调学习速率和迭代次数
- 代码简洁

#### 缺点

- 特征量大的时候计算复杂
- 适用范围较窄

