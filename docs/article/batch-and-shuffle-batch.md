# 详解TensorFlow之tf.train.batch与tf.train.shuffle_batch
tf.train.batch与tf.train.shuffle_batch的作用都是从队列中读取数据，它们的区别是是否随机打乱数据来读取。
## tf.train.batch()函数
```python
tf.train.batch(
    tensors,
    batch_size,
    num_threads=1,
    capacity=32,
    enqueue_many=False,
    shapes=None,
    dynamic_pad=False,
    allow_smaller_final_batch=False,
    shared_name=None,
    name=None
)
```
- tensors：一个列表或字典的tensor用来进行入队
- batch_size：每次从队列中获取出队数据的数量
- num_threads：用来控制入队tensors线程的数量，如果num_threads大于1，则batch操作将是非确定性的，输出的batch可能会乱序
- capacity：一个整数，用来设置队列中元素的最大数量
- enqueue_many：在tensors中的张量是否是单个样本，若为False，则认为tensors代表一个样本．输入张量形状为[x, y, z]时，则输出张量形状为[batch_size, x, y, z]，若为True，则认为tensors代表一批样本，其中第一个维度为样本的索引，并且所有成员tensors在第一维中应具有相同大小．若输入张量形状为[*, x, y, z]，则输出张量的形状为[batch_size, x, y, z]
- shapes：每个样本的shape，默认是tensors的shape
- dynamic_pad：为True时允许输入变量的shape，出队后会自动填补维度，来保持与batch内的shapes相同
- allow_smaller_final_batch：为True队列中的样本数量小于batch_size时，出队的数量会以最终遗留下来的样本进行出队，如果为Flalse，小于batch_size的样本不会做出队处理
- shared_name：如果设置，则队列将在多个会话中以给定名称共享
- name：操作的名字

### 代码演示
```python
#!/usr/bin/python
# coding:utf-8
import tensorflow as tf
import numpy as np

images = np.random.random([5, 2])  # 5x2的矩阵
print(images)
label = np.asarray(range(0, 5))  # [0, 1, 2, 3, 4]
print(label)
# 将数组转换为张量
images = tf.cast(images, tf.float32)
print(images)
label = tf.cast(label, tf.int32)
print(label)
# 切片
input_queue = tf.train.slice_input_producer([images, label], shuffle=False)
# 按顺序读取队列中的数据
image_batch, label_batch = tf.train.batch(input_queue, batch_size=10, num_threads=1, capacity=64)

with tf.Session() as sess:
    # 线程的协调器
    coord = tf.train.Coordinator()
    # 开始在图表中收集队列运行器
    threads = tf.train.start_queue_runners(sess, coord)
    image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
    for j in range(5):
        print(image_batch_v[j]),
        print(label_batch_v[j])
    # 请求线程结束
    coord.request_stop()
    # 等待线程终止
    coord.join(threads)
```
输出
```python
[0.15518834 0.4924818 ]
0
[0.3907916  0.05013292]
1
[0.41328526 0.802318  ]
2
[0.43541858 0.9412442 ]
3
[0.16782863 0.6347318 ]
4
```
参考代码移步[Github](https://github.com/cwyd0822/cifar10-tensorflow-read-write/blob/master/batch.py)

## tf.train.shuffle_batch()函数
```python
tf.train.shuffle_batch(
    tensors,
    batch_size,
    capacity,
    min_after_dequeue,
    num_threads=1,
    seed=None,
    enqueue_many=False,
    shapes=None,
    allow_smaller_final_batch=False,
    shared_name=None,
    name=None
)
```
可以看出，跟tf.train.batch的参数是一样的，只是这里多了个seed和min_after_dequeue，其中seed表示随机数的种子，min_after_dequeue是出队后队列中元素的最小数量，用于确保元素的混合级别，这个参数一定要比capacity小。

### 代码演示
```python
#!/usr/bin/python
# coding:utf-8
import tensorflow as tf
import numpy as np

images = np.random.random([5, 2])
label = np.asarray(range(0, 5))
# 列表转成张量
images = tf.cast(images, tf.float32)
label = tf.cast(label, tf.int32)
# 输入张量列表
input_queue = tf.train.slice_input_producer([images, label], shuffle=False)
# 将队列中数据打乱后再读取出来
image_batch, label_batch = tf.train.shuffle_batch(input_queue, batch_size=10, num_threads=1, capacity=64, min_after_dequeue=1)

with tf.Session() as sess:
    # 线程的协调器
    coord = tf.train.Coordinator()
    # 开始在图表中收集队列运行器
    threads = tf.train.start_queue_runners(sess, coord)
    image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
    for j in range(5):
        # print(image_batch_v.shape, label_batch_v[j])
        print(image_batch_v[j]),
        print(label_batch_v[j])
    # 请求线程结束
    coord.request_stop()
    # 等待线程终止
    coord.join(threads)
```
输出
```python
[0.93350357 0.9003149 ]
0
[0.7407439 0.896775 ]
1
[0.6358515  0.69127077]
2
[0.96927387 0.7181145 ]
3
[0.93350357 0.9003149 ]
0
```
代码已经上传至[Github](https://github.com/cwyd0822/cifar10-tensorflow-read-write/blob/master/shuffle_batch.py)