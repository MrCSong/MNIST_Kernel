import h5py
import numpy as np

f = h5py.File('Weight/MNIST_cov1.h5','r')   #打开h5文件
# 方法一
# -------------
# f.keys()
# print([key for key in f.keys()])
#
# print('first, we get values of conv2d_1:', f['conv2d_1'][:])
# print(f['conv2d_1'][:].shape)
# 方法二
# --------------
# def print_keras_wegiths(weight_file_path):
#     f = h5py.File(weight_file_path)  # 读取weights h5文件返回File类
#     try:
#         if len(f.attrs.items()):
#             print("{} contains: ".format(weight_file_path))
#             print("Root attributes:")
#         for key, value in f.attrs.items():
#             print("  {}: {}".format(key, value))  # 输出储存在File类中的attrs信息，一般是各层的名称
#
#         for layer, g in f.items():  # 读取各层的名称以及包含层信息的Group类
#             print("  {}".format(layer))
#             print("    Attributes:")
#             for key, value in g.attrs.items(): # 输出储存在Group类中的attrs信息，一般是各层的weights和bias及他们的名称
#                 print("      {}: {}".format(key, value))
#
#             # print("    Dataset:")
#             # for name, d in g.items(): # 读取各层储存具体信息的Dataset类
#             #     print("      {}: {}".format(name, d.value.shape)) # 输出储存在Dataset中的层名称和权重，也可以打印dataset的attrs，但是keras中是空的
#             #     print("      {}: {}".format(name. d.value))
#     finally:
#         f.close()

# weight_file_path = 'Weight/MNIST_cov1.h5'
# print_keras_wegiths(weight_file_path)
# -------------------------------------

for key, value in f.attrs.items():
    print("  {}: {}".format(key, value))

group1 = f.get('conv2d_1')
group2 = f.get('conv2d_2')

g1 = group1.get('conv2d_1')
g2 = group2.get('conv2d_2')

w1 = g1.get('kernel:0')
w1 = np.array(w1)
b1 = g1.get('bias:0')
b1 = np.array(b1)
w2 = g2.get('kernel:0')
w2 = np.array(w2)
b2 = g2.get('bias:0')
b2 = np.array(b2)

# print(w1.shape)
# print(w2.shape)

# print(w1)
# print(w2)



# 将权重拍平为一维数组
x = np.reshape(w1,[800,1])    # -1：为了把原来的向量铺平，28*28：图片的长宽，1:颜色通道 （RBG：3，黑白：1）
# print(x)

# 计算方差
arr_mean = np.mean(x)
arr_std = np.std(x,ddof=1)
arr_var = np.var(x)
print("方差为：%f" % arr_var)
print("标准差为:%f" % arr_std)
print("平均值为：%f" % arr_mean)

# 生成高斯噪声，跟权重维度相同
normal_kernel = np.random.normal(-0.017175, 0.05, (5,5,32))
# print(normal_kernel)
x_normal_kernel = np.reshape(normal_kernel,[800,1])
# print(x_normal_kernel)

# 加入高斯噪声
y=x+x_normal_kernel
# print(y)
w_1 = np.reshape(y,[5,5,1,32])
# print(w_1)

f.close()


# hf = h5py.File('Weight/MNIST_cov1.h5','w')
# hf.create_dataset('conv2d_1', data=w_1)
# hf.close()
