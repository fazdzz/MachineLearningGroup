# Week 8 codes

## 说明

这里是B站上第八周（coursera上第四周）作业的代码，具体要求参见ex3.pdf。

## 环境

matlab2016b全部通过在线评测。

## 其他

其中计算θ的时候，Ng本身提供了一个fmincg函数。我查了下这个是Octave的函数，Ng可能是为了兼容性特意带了一个文件。

用这个fmincg函数计算很快，训练集大概1min就出结果了，准确率95%。

我用fminunc重写了一遍，只迭代6次都要等10min，不过准确率96%~~（有什么用呢）~~，实际提交的时候由于测试集很弱可以迭代次数调高一些。