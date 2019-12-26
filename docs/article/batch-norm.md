# 详解Batchnorm
> Batchnorm是深度网络中经常用到的加速神经网络训练，加速收敛速度及稳定性的算法，可以说是目前深度网络必不可少的一部分。
## 为什么Batchnorm
- 机器学习领域有个很重要的假设：IID独立同分布假设，就是假设训练数据和测试数据是满足相同分布的，这是通过训练数据获得的模型能够在测试集获得好的效果的一个基本保障。那BatchNorm的作用是什么呢？BatchNorm就是在深度神经网络训练过程中使得每一层神经网络的输入保持相同分布的。
- 深度学习的话尤其是在CV上都需要对数据做归一化，因为深度神经网络主要就是为了学习训练数据的分布，并在测试集上达到很好的泛化效果，但是，如果我们每一个batch输入的数据都具有不同的分布，显然会给网络的训练带来困难。另一方面，数据经过一层层网络计算后，其数据分布也在发生着变化，此现象称为Internal Covariate Shift，接下来会详细解释，会给下一层的网络学习带来困难。batchnorm直译过来就是批规范化，就是为了解决这个分布变化问题。
- Internal Covariate Shift ：此术语是google小组在论文Batch Normalizatoin 中提出来的，其主要描述的是：训练深度网络的时候经常发生训练困难的问题，因为，每一次参数迭代更新后，上一层网络的输出数据经过这一层网络计算后，数据的分布会发生变化，为下一层网络的学习带来困难（神经网络本来就是要学习数据的分布，要是分布一直在变，学习就很难了），此现象称之为Internal Covariate Shift。之前的解决方案就是使用较小的学习率，和小心的初始化参数，对数据做白化处理，但是显然治标不治本。
- covariate shift：Internal Covariate Shift 和Covariate Shift具有相似性，但并不是一个东西，前者发生在神经网络的内部，所以是Internal，后者发生在输入数据上。Covariate Shift主要描述的是由于训练数据和测试数据存在分布的差异性，给网络的泛化性和训练速度带来了影响，我们经常使用的方法是做归一化或者白化。想要直观感受的话，看下图：
![image](https://imgconvert.csdnimg.cn/aHR0cDovL2FpY2hlbndlaS5vc3MtYXAtc291dGhlYXN0LTEuYWxpeXVuY3MuY29tL2dpdGh1Yi9jaWZhcjEwLWNsYXNzaWZpY2F0aW9uLXRlbnNvcmZsb3ctc2xpbS8yLnBuZw?x-oss-process=image/format,png)
> 举个简单线性分类栗子，假设我们的数据分布如a所示，参数初始化一般是0均值，和较小的方差，此时拟合的y=wx+b
> 如b图中的橘色线，经过多次迭代后，达到紫色线，此时具有很好的分类效果，但是如果我们将其归一化到0点附近，显然会加快训练速度，如此我们更进一步的通过变换拉大数据之间的相对差异性，那么就更容易区分了。
- Covariate Shift 就是描述的输入数据分布不一致的现象，对数据做归一化当然可以加快训练速度，能对数据做去相关性，突出它们之间的分布相对差异就更好了。Batchnorm做到了，前文已说过，Batchnorm是归一化的一种手段，极限来说，这种方式会减小图像之间的绝对差异，突出相对差异，加快训练速度。所以说，并不是在深度学习的所有领域都可以使用BatchNorm。

## Batchnorm的原理
为了降低Internal Covariate Shift带来的影响，其实只要进行归一化就可以的。比如，我们把network每一层的输出都整为方差为1，均值为0的正态分布，这样看起来是可以解决问题，但是想想，network好不容易学习到的数据特征，被你这样一弄又回到了解放前了，相当于没有学习了。所以这样是不行的，大神想到了一个大招：变换重构，引入了两个可以学习的参数γ、β，当然，这也是算法的灵魂所在：
![image](https://img-blog.csdn.net/20180417214133986?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3JlbWFuZW50ZWQ=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
具体的算法流程如下:
![](https://img-blog.csdn.net/20180417214248222?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3JlbWFuZW50ZWQ=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
 Batch Normalization 是对一个batch来进行normalization的，例如我们的输入的一个batch为：β=x_(1...m)，输出为：y_i=BN(x)。具体的完整流程如下：

 1. 求出该batch数据x的均值
![](https://img-blog.csdn.net/20180417214812576?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3JlbWFuZW50ZWQ=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
2. 求出该batch数据的方差
![](https://img-blog.csdn.net/20180417214842955?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3JlbWFuZW50ZWQ=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
3. 对输入数据x做归一化处理，得到：
![](https://img-blog.csdn.net/20180417214947300?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3JlbWFuZW50ZWQ=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
4. 最后加入可训练的两个参数：缩放变量γ和平移变量β，计算归一化后的值：
![](https://img-blog.csdn.net/20180417215236561?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3JlbWFuZW50ZWQ=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) 
加入了这两个参数之后，网络就可以更加容易的学习到更多的东西了。先想想极端的情况，当缩放变量γ和平移变量β分别等于batch数据的方差和均值时，最后得到的yi就和原来的xi一模一样了，相当于batch normalization没有起作用了。这样就保证了每一次数据经过归一化后还保留的有学习来的特征，同时又能完成归一化这个操作，加速训练。

    引入参数的更新过程，也就是链式法则：
![](https://img-blog.csdn.net/20180417220023946?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3JlbWFuZW50ZWQ=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
### 一个简单的例子：
```python
def Batchnorm_simple_for_train(x, gamma,beta, bn_param):
"""
param:x   : 输入数据，设shape(B,L)
param:gama : 缩放因子  γ
param:beta : 平移因子  β
param:bn_param   : batchnorm所需要的一些参数
   eps      : 接近0的数，防止分母出现0
   momentum : 动量参数，一般为0.9，0.99， 0.999
   running_mean ：滑动平均的方式计算新的均值，训练时计算，为测试数据做准备
   running_var  : 滑动平均的方式计算新的方差，训练时计算，为测试数据做准备
"""
   running_mean = bn_param['running_mean'] #shape = [B]
   running_var = bn_param['running_var']   #shape = [B]
   results = 0. # 建立一个新的变量
   x_mean=x.mean(axis=0)  # 计算x的均值
   x_var=x.var(axis=0)    # 计算方差
   x_normalized=(x-x_mean)/np.sqrt(x_var+eps)       # 归一化
   results = gamma * x_normalized + beta            # 缩放平移
   running_mean = momentum * running_mean + (1 - momentum) * x_mean
   running_var = momentum * running_var + (1 - momentum) * x_var    #记录新的值
   bn_param['running_mean'] = running_mean
   bn_param['running_var'] = running_var   
   return results , bn_param
```
看完这个代码是不是对batchnorm有了一个清晰的理解，首先计算均值和方差，然后归一化，然后缩放和平移，完事！但是这是在训练中完成的任务，每次训练给一个批量，然后计算批量的均值方差，但是在测试的时候可不是这样，测试的时候每次只输入一张图片，这怎么计算批量的均值和方差，于是，就有了代码中下面两行，在训练的时候实现计算好mean var测试的时候直接拿来用就可以了，不用计算均值和方差。
```python
running_mean = momentum * running_mean + (1- momentum) * x_mean
running_var = momentum * running_var + (1 -momentum) * x_var
```
所以，测试的时候是这样的：
```python
def Batchnorm_simple_for_test(x, gamma,beta, bn_param):
"""
param:x   : 输入数据，设shape(B,L)
param:gama : 缩放因子  γ
param:beta : 平移因子  β
param:bn_param   : batchnorm所需要的一些参数
   eps      : 接近0的数，防止分母出现0
   momentum : 动量参数，一般为0.9，0.99， 0.999
   running_mean ：滑动平均的方式计算新的均值，训练时计算，为测试数据做准备
   running_var  : 滑动平均的方式计算新的方差，训练时计算，为测试数据做准备
"""
   running_mean = bn_param['running_mean'] #shape = [B]
   running_var = bn_param['running_var']   #shape = [B]
   results = 0. # 建立一个新的变量
   x_normalized=(x-running_mean )/np.sqrt(running_var +eps)       # 归一化
   results = gamma * x_normalized + beta            # 缩放平移
   return results , bn_param
```
## Batch Normalization的带来的优势
- 没有它之前，需要小心的调整学习率和权重初始化，但是有了BN可以放心的使用大学习率，但是使用了BN，就不用小心的调参了，较大的学习率极大的提高了学习速度，

- Batchnorm本身上也是一种正则的方式，可以代替其他正则方式如dropout等

- 另外，个人认为，batchnorm降低了数据之间的绝对差异，有一个去相关的性质，更多的考虑相对差异性，因此在分类任务上具有更好的效果

## 参考文章：
- [《Batch Normalization 和 Group Normalization》](https://blog.csdn.net/remanented/article/details/79980486)
