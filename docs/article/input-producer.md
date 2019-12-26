# 详解TensorFlow之tf.train.slice_input_producer和tf.train.string_input_producer生成器
## tf.train.slice_input_producer是什么
tf.train.slice_input_producer是一个tensor生成器，作用是按照设定，每次从一个tensor列表中按顺序或者随机抽取出一个tensor放入队列。 

## tf.train.slice_input_producer函数
```python
slice_input_producer(tensor_list, num_epochs=None, shuffle=True, seed=None, capacity=32, shared_name=None, name=None)
```
- tensor_list：包含一系列tensor的列表，表中tensor的第一维度的值必须相等，即个数必须相等，有多少个图像，就应该有多少个对应的标签。
- 第二个参数num_epochs: 可选参数，是一个整数值，代表迭代的次数，如果设置 num_epochs=None,生成器可以无限次遍历tensor列表，如果设置为 num_epochs=N，生成器只能遍历tensor列表N次。 
- 第三个参数shuffle：bool类型，设置是否打乱样本的顺序。一般情况下，如果shuffle=True，生成的样本顺序就被打乱了，在批处理的时候不需要再次打乱样本，使用 tf.train.batch函数就可以了;如果shuffle=False,就需要在批处理时候使用 tf.train.shuffle_batch函数打乱样本。
- 第四个参数seed: 可选的整数，是生成随机数的种子，在第三个参数设置为shuffle=True的情况下才有用。
- 第五个参数capacity：设置tensor列表的容量。
- 第六个参数shared_name：可选参数，如果设置一个‘shared_name’，则在不同的上下文环境（Session）中可以通过这个名字共享生成的tensor。
- 第七个参数name：可选，设置操作的名称。

## tf.train.start_queue_runners()函数
- TensorFlow的Session对象是支持多线程的，可以在同一个会话（Session）中创建多个线程，并行执行。在Session中的所有线程都必须能被同步终止，异常必须能被正确捕获并报告，会话终止的时候，队列必须能被正确地关闭。 
- TensorFlow提供了两个类来实现对Session中多线程的管理：tf.Coordinator和 tf.QueueRunner，这两个类往往一起使用。
- Coordinator类用来管理在Session中的多个线程，可以用来同时停止多个工作线程并且向那个在等待所有工作线程终止的程序报告异常，该线程捕获到这个异常之后就会终止所有线程。使用tf.train.Coordinator()来创建一个线程管理器（协调器）对象。
- QueueRunner类用来启动tensor的入队线程，可以用来启动多个工作线程同时将多个tensor（训练数据）推送入文件名称队列中，具体执行函数是 tf.train.start_queue_runners ， 只有调用tf.train.start_queue_runners 之后，才会真正把tensor推入内存序列中，供计算单元调用，否则会由于内存序列为空，数据流图会处于一直等待状态。

## 代码演示：reader_cifar10-1.py
```python
import tensorflow as tf

# 定义4个图片路径列表
images = ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg']
# 定义4个Label的列表
labels = [1, 2, 3, 4]

# 产生图像和标签对应的tensor
[images, labels] = tf.train.slice_input_producer([images, labels],
                              num_epochs=2,
                              shuffle=True)

with tf.Session() as sess:
    # 对全局的变量进行初始化
    sess.run(tf.local_variables_initializer())
    # 放入线程后，才会产生张量
    tf.train.start_queue_runners(sess=sess)  # 启动队列填充的线程

    for i in range(8):  # 从文件队列中获取数据， 总共4条 * 2epoch，所以最多取8条
        print(sess.run([images, labels]))
```
输出如下
```python
[b'image1.jpg', 1]
[b'image4.jpg', 4]
[b'image3.jpg', 3]
[b'image2.jpg', 2]
[b'image2.jpg', 2]
[b'image1.jpg', 1]
[b'image4.jpg', 4]
[b'image3.jpg', 3]
```
代码已经上传到[Github](https://github.com/cwyd0822/cifar10-tensorflow-read-write/blob/master/reader_cifar10-1.py)

## tf.train.string_input_producer()函数
这个函数跟tf.train.slice_input_producer()类似，不过是针对文件的生成器，传入文件路径列表，每次吐出一个文件。演示代码如下：
```python
"""
读取文件数据
"""

import tensorflow as tf

# 我们放了3个文件在相应的位置
filename = ['data/A.csv', 'data/B.csv', 'data/C.csv']

# 将文件的路径作为参数传入函数
# 输出是文件队列，无法直接获取文件的值
file_queue = tf.train.string_input_producer(filename,
                                            shuffle=True,
                                            num_epochs=2)

# 文件读取器
reader = tf.WholeFileReader()
# key：文件名 value:文件值
key, value = reader.read(file_queue)

with tf.Session() as sess:
    # 对局部变量进行赋值
    sess.run(tf.local_variables_initializer())
    tf.train.start_queue_runners(sess=sess)  # 定义文件队列填充的线程
    for i in range(6):  # 文件数量3 * 2epochs
        print(sess.run([key, value]))
```
输出如下结果：
```python
[b'data/A.csv', b'1.jpg,1\n2.jpg,2\n3.jpg,3\n']
[b'data/B.csv', b'4.jpg,4\n5.jpg,5\n6.jpg,6\n']
[b'data/C.csv', b'7.jpg,7\n8.jpg,8\n9.jpg,9\n']
[b'data/B.csv', b'4.jpg,4\n5.jpg,5\n6.jpg,6\n']
[b'data/A.csv', b'1.jpg,1\n2.jpg,2\n3.jpg,3\n']
[b'data/C.csv', b'7.jpg,7\n8.jpg,8\n9.jpg,9\n']
```
这部分代码也放到了我的[Github](https://github.com/cwyd0822/cifar10-tensorflow-read-write/blob/master/reader_cifar10-2.py)上，大家可以参考。
