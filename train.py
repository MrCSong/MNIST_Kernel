"""
1.MNIST network and train it.
2.read the convolution kernal
3.calculate each samples' WD


8:40
"""

import random

from keras.models import Sequential, load_model
from keras.layers import Convolution2D, Dense, Flatten, Activation, MaxPooling2D, core,GaussianNoise
from keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
y_test = to_categorical(y_test, 10)
y_train = to_categorical(y_train, 10)


def AddGassisNoise_0(weight):
    # 将权重拍平为一维数组
    weight_Dense = weight
    x = np.reshape(weight_Dense, [800, 1])
    print(x)

    # 计算方差
    arr_mean = np.mean(x)
    arr_std = np.std(x, ddof=1)
    arr_var = np.var(x)
    print("方差为：%f" % arr_var)
    print("标准差为:%f" % arr_std)
    print("平均值为：%f" % arr_mean)

    # 生成高斯噪声，跟权重维度相同
    i = 0.047
    while i > 0:
        print(i)
        normal_kernel = np.random.normal(0.001894, i, (5, 5, 1, 32))
        print(normal_kernel)
        x_normal_kernel = np.reshape(normal_kernel, [800, 1])
        print(x_normal_kernel)

        # 加入高斯噪声
        y = x + x_normal_kernel
        np.savetxt("Weight/cov1_"+str(i)+".csv", y, delimiter=",")
        # print(y)
        weight_Dense_0_change = np.reshape(y, [5, 5, 1, 32])
        i-=0.001
        i = round(i, 3)

    return weight_Dense_0_change


def AddGassisNoise_1(weight):

    # 将权重拍平为一维数组
    weight_Dense = weight
    x = np.reshape(weight_Dense, [18432, 1])
    print(x)

    # 计算方差
    arr_mean = np.mean(x)
    arr_std = np.std(x, ddof=1)
    arr_var = np.var(x)
    print("方差为：%f" % arr_var)
    print("标准差为:%f" % arr_std)
    print("平均值为：%f" % arr_mean)

    # 生成高斯噪声，跟权重维度相同
    normal_kernel = np.random.normal(-0.000090, 0.008, (3, 3, 32, 64))
    print(normal_kernel)
    x_normal_kernel = np.reshape(normal_kernel, [18432, 1])
    print(x_normal_kernel)

    # 加入高斯噪声
    y = x + x_normal_kernel
    np.savetxt("Weight/cov1_" + str(i) + ".csv", y, delimiter=",")
    # print(y)
    weight_Dense_0_change = np.reshape(y, [3, 3, 32, 64])
    return weight_Dense_0_change


def AddGassisNoise_2(weight):

    # 将权重拍平为一维数组
    weight_Dense = weight
    x = np.reshape(weight_Dense, [73728, 1])  # -1：为了把原来的向量铺平，28*28：图片的长宽，1:颜色通道 （RBG：3，黑白：1）
    print(x)

    # 计算方差
    arr_mean = np.mean(x)
    arr_std = np.std(x, ddof=1)
    arr_var = np.var(x)
    print("方差为：%f" % arr_var)
    print("标准差为:%f" % arr_std)
    print("平均值为：%f" % arr_mean)

    # 生成高斯噪声，跟权重维度相同
    normal_kernel = np.random.normal(-0.000111, 0.004, (3, 3, 64, 128))
    print(normal_kernel)
    x_normal_kernel = np.reshape(normal_kernel, [73728, 1])
    print(x_normal_kernel)

    # 加入高斯噪声
    y = x + x_normal_kernel
    # np.savetxt("Weight/cov3_" + str(i) + ".csv", y, delimiter=",")
    # print(y)
    weight_Dense_0_change = np.reshape(y, [3, 3, 64, 128])
    return weight_Dense_0_change


model = Sequential()
model.add(Convolution2D(32, (5, 5), input_shape=(28, 28, 1)))
model.add(MaxPooling2D(2, 2))
model.add(Activation('relu'))

model.add(Convolution2D(64, (3, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Activation('relu'))

model.add(Convolution2D(128, (3, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))



# for i in range(0,11):
#     x=model.get_layer(index=i).output
#     print(x)

model.summary();
for layer in model.layers:
    print(layer.name)
print(model.get_config())


# design model
def train():
    adam = Adam(lr=0.001)
    # compile model
    model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
    # ********************
    # load weight
    model.load_weights('Weight/MNIST_1.h5')

    # ***************************************************************
    weight_Dense,bias_Dense = model.get_layer(index=0).get_weights()
    print(weight_Dense.shape)
    print(weight_Dense)
    # ***************************************************************
    # weight_Cov_0_change = AddGassisNoise_0(weight_Dense)
    # weight_Cov_1_change = AddGassisNoise_1(weight_Dense)
    # weight_Cov_2_change = AddGassisNoise_2(weight_Dense)

    # ***************************************************************
    # 将权重拍平为一维数组
    x = np.reshape(weight_Dense, [800, 1])  # -1：为了把原来的向量铺平，28*28：图片的长宽，1:颜色通道 （RBG：3，黑白：1）
    # print(x)

    # 计算方差
    arr_mean = np.mean(x)
    arr_std = np.std(x, ddof=1)
    arr_var = np.var(x)
    print("方差为：%f" % arr_var)
    print("标准差为:%f" % arr_std)
    print("平均值为：%f" % arr_mean)

    # 生成高斯噪声，跟权重维度相同
    list = []
    i = 0.047
    while i > 0:
        # 生成高斯噪声，跟权重维度相同
        normal_kernel = np.random.normal(0, i, (5, 5, 1, 32))
        # print(normal_kernel)
        x_normal_kernel = np.reshape(normal_kernel, [800, 1])
        print(x_normal_kernel)

        # 加入高斯噪声
        y = x + x_normal_kernel
        np.savetxt("Weight/cov1_" + str(i) + ".csv", y, delimiter=",")
        # print(y)
        weight_Cov_0_change = np.reshape(y, [5, 5, 1, 32])
        # ***********************************************************
        model.layers[0].set_weights([weight_Cov_0_change,bias_Dense]) # MUST ADD []Brackets!
        # print(weight_Cov_0_change.shape)
        # ************************************************************

        # model.fit(x_train, y_train, batch_size=100, epochs=1)
        # test model

        loss, accuracy = model.evaluate(x_test, y_test, batch_size=100)
        # print(accuracy)
        m=accuracy

        list.append(m)
        # print(m)
        i -= 0.00001
        i = round(i, 5)

    np.savetxt("Weight/accuracy.csv", list, delimiter=",")
    # print("**************")
    # print(list)

    # loss, accuracy = model.evaluate(x_test, y_test, batch_size=100)
    # print(accuracy)
    # save model
    # model.save('Weight/MNIST_1.h5')
    # model.save_weights('Weight/MNIST_cov1.h5') # difference from save


#############################################################
"""
I know there is one reason that why I could show the pic.
I must separate this two code. One is train, the other is predict. 
In the train code, the train img has been reshape the dimension,
so if you still use the reshape dimension to do the plt.imshow, it can't 
succeed. So when I comment out the test img reshape operation, I succeed!
So, Do not use the img that has been reshaped.
"""

##############################################################
def test():
    # 显示该数字
    # x = x_test[100]
    # 随机数
    # index = random.randint(0, x_test.shape[0])
    # print(index)

    # plt.subplot(224)
    # plt.imshow(x_train[4545]) ##wrong way, you can't use the changed image.
    # plt.show()
    x = x_test[110]
    y = y_test[110]

    # plt.imshow(x, cmap='gray_r')
    # plt.title("original {}".format(y))
    # plt.show()

    model.load_weights('Weight/MNIST_1.h5')
    k = np.array(x_test[100])
    print(k.shape)
    y= k.reshape(1,28,28,1)
    print(y.shape)

    prediction = model.predict(y)
    print(prediction)


    class_pred = model.predict_classes(y)
    print(class_pred)


if __name__ == '__main__':
    train()
    # test()
