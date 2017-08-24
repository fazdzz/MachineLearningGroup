# Generalized Linear Models

## 简介

Generalized Linear Models(GLM)就是广义线性模型，之前在监督学习中学到的回归问题和分类问题的算法，其实本质上都可以通过GLM推导出来，这里**简单**介绍一下，主要还是为了搭建一个知识体系。

如果要更加准确的解释，还请参考[讲义](http://cs229.stanford.edu/notes/cs229-notes1.pdf)。

## 定义(exponential family distributions)

讲义中给出的定义

![](http://latex.codecogs.com/gif.latex?p%28y%3B%5Ceta%29%20%3D%20b%28y%29e%5E%7B%5Ceta%5E%7BT%7DT%28y%29-a%28%5Ceta%29%7D)

同时讲义中也提到更加完整的定义应该是，不过这里为了方便仅用上面的简化定义。

![](http://latex.codecogs.com/gif.latex?p%28y%3B%5Ceta%2C%5Ctau%29%20%3D%20b%28y%2C%20%5Ctau%29e%5E%7B%5Cfrac%20%7B%5Ceta%5E%7BT%7DT%28y%29-a%28%5Ceta%29%7D%7Bc%28%5Ctau%29%7D%7D)

注：这里原讲义中可能出现了一点小笔误，已经按照wiki修正。

## 伯努利分布和高斯分布

其实都是上述GLM的特例。

### 伯努利分布

取值如下

![](http://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20%26%20%5Ceta%20%3D%20log%28%5Cphi/%281-%5Cphi%29%29%5C%5C%20%26%20T%28y%29%20%3D%20y%5C%5C%20%26%20a%28%5Ceta%29%20%3D%20-%20log%281-%5Cphi%29%5C%5C%20%26%20b%7By%7D%20%3D%201%20%5Cend%7Balgned%7D)

### 高斯分布

取值如下

![](http://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20%26%20%5Ceta%20%3D%20%5Cmu%5C%5C%20%26%20T%28y%29%20%3D%20y%5C%5C%20%26%20a%28%5Ceta%29%20%3D%20%5Cmu%5E%7B2%7D/2%5C%5C%20%26%20b%7By%7D%20%3D%20%281/%5Csqrt%7B2%5Cpi%7D%29e%5E%7B-y%5E%7B2%7D/2%7D%20%5Cend%7Balgned%7D)

当然这里设σ^2=1，不然就要用完整定义了。

## 联系

有了GLM怎么推导出我们的学习算法呢？

> 1. y | x; θ ∼ ExponentialFamily(η). I.e., given x and θ, the distribution of y follows some exponential family distribution, with parameter η.
> 2. Given x, our goal is to predict the expected value of T (y) given x. In most of our examples, we will have T (y) = y, so this means we would like the prediction h(x) output by our learned hypothesis h to satisfy h(x) = E[y|x]. (Note that this assumption is satisfied in the choices for hθ(x) for both logistic regression and linear regression. For instance, in logistic regression, we had hθ(x) = p(y = 1|x; θ) = 0 · p(y = 0|x; θ) + 1 · p(y = 1|x; θ) = E[y|x; θ].)
> 3. The natural parameter η and the inputs x are related linearly: η = θT x. (Or, if η is vector-valued, then ηi = θi T x.) 

简单来说就是

1. 选择一个适当的模型（之前线性回归应用的是高斯分布，逻辑回归应用的是伯努利分布，具体细节参考讲义）
2. 在实际学习过程中，我们一般是用x来预测T(y)（一般T(y)=y），而我们的假设h(x)就是选择模型的数学期望E[y|x]。
3. 在选择η的时候一般为η=θ^TX。

同时讲义中给出了第三点的解释

> The third of these assumptions might seem the least well justified of the above, and it might be better thought of as a “design choice” in our recipe for designing GLMs, rather than as an assumption per se. These three assumptions/design choices will allow us to derive a very elegant class of learning algorithms, namely GLMs, that have many desirable properties such as ease of learning. Furthermore, the resulting models are often very effective for modelling different types of distributions over y; for example, we will shortly show that both logistic regression and ordinary least squares can both be derived as GLMs 

有了这三条规则，我们就可以用合理的模型来直接构造出相应的学习算法。

## 误差函数

之前已经提到了如何根据GLM构造学习算法，但是误差函数又是如何制定的呢？

根据第一条，P(y|x)服从某个分布，也就是说我们有对于**每组数据预测正确的概率**。

于是我们有所有数据预测正确的可能性P=P1*P2...Pm，我们只需要P最大即可，代价函数就是这么选择出来的。

（个人理解，以后若是有更严谨的理解方式再更新）