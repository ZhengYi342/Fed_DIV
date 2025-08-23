# Large amount of credit goes to:
# https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan_gp/wgan_gp.py and
# https://github.com/eriklindernoren/Keras-GAN/blob/master/cgan/cgan.py
# which I've used as a reference for this implementation
# Author: Hanling Wang
# Date: 2018-11-21

from __future__ import print_function, division

import sys

import pandas as pd

import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.utils import shuffle
from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Embedding,LayerNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop,Adam
from functools import partial
import os
import random

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


import keras.backend as K

import matplotlib.pyplot as plt

import math

import numpy as np

# feature = 29
# name="creditcard"

feature = 6
name = "FraudDetection"
#记得改判别器和生成器输入的维度
def eucliDist(A,B):
    return np.sqrt(sum(np.power((A - B), 2)))
class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""

    def _merge_function(self, inputs):
        global batch_size             #批次大小
        alpha = K.random_uniform((batch_size, 1, 1, 1))#使用K.random_uniform函数生成一个形状为(batch_size, 1, 1, 1)的随机数矩阵alpha，用于表示加权平均的权重
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])#返回一个加权平均样本（插值样本），将真实样本和生成样本按照alpha的权重进行加权平均


class WGANGP():
    def __init__(self, epochs=100, batch_size=32, sample_interval=1):
        self.img_rows = 1                                             #样本数
        self.img_cols =feature                                              #特征数
        self.channels = 1                                             #通道数
        self.nclasses = 2                                             #类别数
        self.img_shape = (self.img_rows, self.img_cols, self.channels)#输入样本的形状
        self.latent_dim = 70                                          #映射的维度
        self.losslog = []                                             #损失列表
        self.epochs = epochs                                          #训练轮次
        self.batch_size = batch_size                                  #每次取的样本数
        self.sample_interval = sample_interval                        #存参数的间隔

        # Following parameter and optimizer set as recommended in paper
        self.n_discriminator = 5                                      #判别器训练次数
        discriminator_optimizer = Adam(lr=0.0001, beta_1=0, beta_2=0.9)  # 优化器（优化模型参数）设置为RMSprop（自适应优化器）
        genger_optimizer = Adam(lr=0.0002)

        # Build the generator and discriminator
        self.generator = self.build_generator()                       #创建生成器和判别器的实例
        self.discriminator = self.build_discriminator()

        # -------------------------------
        # Construct Computational Graph
        #       for the Discriminator
        # -------------------------------

        # Freeze generator's layers while training discriminator
        self.generator.trainable = False                              #在训练判别器时冻结生成器模型的参数，使其在训练过程中不会被更新。

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)                        #定义一个输入层，用于接收真实样本

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))                      #定义一个输入层，用于接收噪声
        # print(z_disc.shape)
        # Generate image based of noise (fake sample) and add label to the input
        #label = Input(shape=(1,))
        # print(label.shape)
        fake_img = self.generator(z_disc)                             #将噪声映射为假样本

        # Discriminator determines validity of the real and fake images
        fake = self.discriminator(fake_img)                           #判别器判别真样本与假样本
        valid = self.discriminator(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])#生成插值样本

        # Determine validity of weighted sample
        validity_interpolated = self.discriminator(interpolated_img) #使用判别器评估插值样本

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  averaged_samples=interpolated_img)  #使用偏函数（partial）来定义梯度惩罚损失函数partial_gp_loss。这个损失函数用于计算梯度惩罚
        partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names

        self.discriminator_model = Model(inputs=[real_img, z_disc], outputs=[valid, fake, validity_interpolated])#模型的输入是真实样本和噪声，输出是真实样本、生成样本和插值样本的评估结果
        self.discriminator_model.compile(loss=[self.wasserstein_loss,
                                        self.wasserstein_loss,
                                        partial_gp_loss],
                                  optimizer=discriminator_optimizer,
                                  loss_weights=[1, 1, 10])                      #编译判别器模型（真实样本的Wasserstein距离损失、生成样本的Wasserstein距离损失、梯度惩罚损失）
        # -------------------------------
        # Construct Computational Graph
        #         for Generator
        # -------------------------------

        # For the generator we freeze the critic's layers
        self.discriminator.trainable = False
        self.generator.trainable = True                                     #冻结判别器参数，解冻生成器参数

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,))                             #定义一个输入层，接收随机噪声
        # add label to the input
        # label = Input(shape=(1,))
        # Generate images based of noise
        img = self.generator(z_gen)                                         #把噪声映射为生成样本
        # Discriminator determines validity
        valid = self.discriminator(img)                                     #评估生成样本
        # Defines generator model
        self.generator_model = Model(z_gen, valid)                          #生成器模型（输入为噪声，输出为生成样本的评估结果）
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=genger_optimizer)  #编译生成器模型，损失函数为生成样本的wasserstein距离损失

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):       #定义梯度惩罚损失函数
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]#求判别器输出对于插值样本的梯度，取第一个梯度
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)#对梯度求平方
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))#对梯度张量的所有维度进行求和，除了第一个维度
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)#对梯度平方和进行开方操作，得到每个样本的梯度的L2范数（梯度的大小）
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)#求梯度惩罚项的值（L2范数减1再平方）
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)#将这批次样本的梯度惩罚项相加并求平均，得到了整个批次样本的梯度惩罚损失

    def wasserstein_loss(self, y_true, y_pred):

        # print(K.mean(y_true * y_pred))

        return K.mean(y_true * y_pred)#算 y_true 和 y_pred 的点积的平均值，即真实标签的平均减去预测标签的平均

    def build_generator(self):
        model = Sequential()  # 定义模型

        # model.add(Dense(256, input_dim=self.latent_dim, activation='relu'))     #添加一个全连接层，输入维度为self.latent_dim，输出维度为256
        # model.add(Dense(1*29*256, activation="relu",input_dim=self.latent_dim))
        model.add(Dense(1 * 6 * 512, input_dim=self.latent_dim))

        # model.add(Activation('relu'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((1, 6, 512)))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        # model.add(Activation('relu'))  # 添加一个LeakyReLU激活函数
        model.add(LeakyReLU(alpha=0.2))

        # model.add(Dropout(0.4))

        # model.add(Dense(512, activation='relu'))                                #添加一个全连接层，输出维度为512
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        # model.add(Activation('relu'))  # 添加一个LeakyReLU激活函数
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(self.channels, kernel_size=3, padding="same",
                         activation='tanh'))  # 添加一个全连接层，输出维度为样本的元素个数，激活函数为’relu’。

        noise = Input(shape=(self.latent_dim,))  # 定义噪声张量

        model_input = noise  # 定义输入层
        img = model(model_input)  # 生成样本

        return Model(noise, img)  # 返回模型

    def build_discriminator(self):
        model = Sequential()

        model.add(Dense(1 * 6 * 128, input_dim=np.prod(self.img_shape)))
        model.add(Reshape((1, 6, 128)))

        model.add(LayerNormalization())

        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(LayerNormalization())
        model.add(LeakyReLU(alpha=0.2))  # 添加一个LeakyReLU激活函数

        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(LayerNormalization())  # 添加一个全连接层，输出维度为512
        model.add(LeakyReLU(alpha=0.2))  # 添加一个LeakyReLU激活函数

        model.add(Conv2D(512, kernel_size=3, padding="same"))
        model.add(LayerNormalization())
        model.add(LeakyReLU(alpha=0.2))  # 添加一个LeakyReLU激活函数

        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(1))  # 添加一个全连接层，输出维度为1，激活函数为’sigmoid’

        img = Input(shape=self.img_shape)  # 定义样本张量

        flat_img = Flatten()(img)
        model_input = flat_img  # 定义输入层

        validity = model(model_input)  # 生成评估结果

        return Model(img, validity)  # 返回模型

    def data_init(self,j):
        dataframe1 = pd.read_csv("D:/all-train/" + str(j) + "-fold/" + name + "_train.csv", header=None)
        dataframe2 = pd.read_csv("D:/all-train/" + str(j) + "-fold/" + name + "_label.csv", header=None)
        l = len(dataframe2.columns) - 1
        m = dataframe2[l].value_counts()[0]
        w = dataframe2[l].value_counts()[1]
        q = dataframe2[l].value_counts()[1]
        dataframe2.rename(columns={0: len(dataframe1.columns)}, inplace=True)
        df = pd.concat([dataframe1, dataframe2], axis=1)
        df2 = df[df[len(dataframe1.columns)] == 1]  # 少数类样本
        df3 = df[df[len(dataframe1.columns)] == 0]  # 多数类样本
        minset = df2.drop([len(dataframe1.columns)], axis=1)
        minset.to_csv('min.csv', header=False, index=False, mode='a')  # 保存为csv文件
        return w,m,q
    def train(self, j ,q):

        dataframe1 = pd.read_csv("D:/all-train/" + str(j) + "-fold/" + name + "_train.csv", header=None)
        dataframe2 = pd.read_csv("D:/all-train/" + str(j) + "-fold/" + name + "_label.csv", header=None)
        l = len(dataframe2.columns) - 1
        m = dataframe2[l].value_counts()[0]
        dataframe2.rename(columns={0: len(dataframe1.columns)}, inplace=True)
        df = pd.concat([dataframe1, dataframe2], axis=1)
        dfmin = df[df[len(dataframe1.columns)] == 1]                           #原始少数类样本
        df3 = df[df[len(dataframe1.columns)] == 0]                            #多数类样本
        dataframe3 = pd.read_csv("D:/pythonProject2/min.csv", header=None)      #更新的少数类样本
        df2 = dataframe3
        dataframe3.loc[:, len(dataframe1.columns)] = 1
        w = int(df2.shape[0])
    #计算信息值
        data_all_class = np.concatenate((dataframe3.values, df3.values))  # 所有数据（带新生成的少数类），带类别
        data_min_class = dfmin.values                                     # 少数类数据(原始，不带新生成的少数类)，带类别
        data_all = data_all_class[:, 0:len(dataframe1.columns)].astype(np.float32)  # 所有数据（带新生成的少数类）
        data_min = data_min_class[:, 0:len(dataframe1.columns)].astype(np.float32)  # 少数类数据(原始，不带新生成的少数类)
        # print(data_min[0][len(dataframe1.columns)])
        dist = np.zeros((data_all.shape[0] - 1, 2))  # 存储距离的数组
        IX = [0.0] * data_min.shape[0]  # 存储信息值的列表
        N_k = 5  # 所指定的近邻数
        for g in range(data_min.shape[0]):
            count = 0
            for h in range(data_all.shape[0]):
                if g != h:
                    dist_temp = eucliDist(data_min[g], data_all[h])
                    dist[count][0] = dist_temp
                    dist[count][1] = data_all_class[h][len(dataframe1.columns)]
                    count += 1
            dist_sort = dist[np.argsort(dist[:, 0])]
            Dall = 0.0
            Dmaj = 0.0
            Nmaj = 0
            for h in range(N_k):  # k个近邻
                if dist_sort[h][1] == 0:
                    Nmaj += 1
                    Dmaj += dist_sort[h][0]
                Dall += dist_sort[h][0]
            if Dmaj == 0.0 or Dall == 0.0:
                DX = 0.0
            else:
                DX = Dmaj / Dall
            CX = Nmaj / N_k
            IX[g] = DX + CX
        sum = 0.0
        for g in range(len(IX)):
            sum += IX[g]
        if sum == 0.0:
            print("无可选的边界样本")
            IX = [1 / data_min.shape[0]] * data_min.shape[0]
        else:
            for g in range(len(IX)):
                IX[g] = IX[g] / sum
        print("信息值计算完毕")
        index_ord=np.arange(data_min.shape[0])
    #训练9个基分类器
        xset = dataframe3.values  # 选出少数类样本
        m_t = df3.values
        W = [1 / w] * w  # 原始权重
        a = [0] * 9      # 分类器重要性
        index = np.arange(w)
        model = ['model1', 'model2', 'model3', 'model4', 'model5', 'model6', 'model7', 'model8', 'model9']

        for b in range(9):
            while 1:
                idx1 = np.random.randint(0, m, w)  # 生成随机序号数组
                idx2 = np.random.choice(index, size=w, replace=True, p=W)  # 按概率随机选择样本

                m_tr = m_t[idx1]  # 选出随机的多数类样本
                mi_tr = xset[idx2]  # 选出随机的少数类样本
                x_l = np.concatenate((m_tr, mi_tr))
                x_tr = x_l[:, 0:len(dataframe1.columns)].astype(np.float32)
                y_tr = x_l[:, len(dataframe1.columns)].astype(np.float32)

                if b == 0:
                    model[b] = GaussianNB()
                elif b == 1:
                    model[b] = BernoulliNB(alpha=1, binarize=0.5)
                elif b == 2:
                    model[b] = KNeighborsClassifier(n_neighbors=6)
                elif b == 3:
                    model[b] = LogisticRegression(penalty='l2', max_iter=10000)
                elif b == 4:
                    model[b] = RandomForestClassifier(n_estimators=8)
                elif b == 5:
                    model[b] = tree.DecisionTreeClassifier()
                elif b == 6:
                    model[b] = GradientBoostingClassifier(n_estimators=200)
                elif b == 7:
                    model[b] = SVC(kernel='rbf', probability=True, C=10, gamma=0.1)
                else:
                    model[b] = MLPClassifier(hidden_layer_sizes=(128,), learning_rate="adaptive", activation='relu',
                                             solver='sgd',
                                             alpha=0.0001, max_iter=10000)

                model[b].fit(x_tr, y_tr)
                y = xset[:, len(dataframe1.columns)].astype(np.float32)  # 原始类别
                x = xset[:, 0:len(dataframe1.columns)].astype(np.float32)  # 原始少数类
                y_pred = model[b].predict(x)
                e = 0.0
                for i in range(w):
                    z = 1
                    if y[i] == y_pred[i]:
                        z = 0
                    e = e + (W[i] * z)
                print('model' + str(b) + '的错误率是' + str(e))
                if e >= 0.5:
                    W = [1 / w] * w
                    continue
                if e == 0.0:
                    e=sys.float_info.min
                a[b] = (1 / 2) * math.log((1 - e) / e)
                su = 0.0
                for i in range(w):
                    z = 1
                    if y[i] == y_pred[i]:
                        z = -1
                    W[i] = W[i] * math.exp(z * a[b])
                    su += W[i]
                W = [x / su for x in W]
                break
            print("基分类器model" + str(b) + "训练完毕\n")
        #分类难度越小，概率越大
        # R_W = [0] * w
        # sum = 0
        # for i in range(len(W)):
        #     R_W[i] = 1 / W[i]
        #     sum += R_W[i]
        # for i in range(len(R_W)):
        #     R_W[i] = R_W[i] / sum
        #训练WGAN-GP

        dataset = df2.values
        X_train = dataset[:, 0:len(dataframe1.columns)].astype(np.float32)   #将特征数组转换为浮点型
        # print(X_train.shape)
        X_train = np.expand_dims(X_train, axis=2)
        X_train = np.expand_dims(X_train, axis=1)                            #将数组扩展为(样本数, 1, 特征数, 1)的形状以适应模型的输入



        valid = -np.ones((self.batch_size, 1))                                 #存储真实样本评估结果，初始全为-1
        fake = np.ones((self.batch_size, 1))                                    #存储生成样本评估结果，初始全为1
        dummy = np.zeros((self.batch_size, 1))  # Dummy gt for gradient penalty  #存储差值样本评估结果，初始全为0
        for epoch in range(self.epochs):
            for _ in range(self.n_discriminator):#先训练判别器

                #idx=np.random.choice(index, size=self.batch_size, replace=True, p=W) #按概率随机选择样本
                idx = random.choices(index_ord, weights=IX, k=self.batch_size)
                #idx = index_np[:self.batch_size]
                imgs = X_train[idx]                   #根据序号选择样本与标签

                noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))#从正态分布的噪声中选一批次噪声，数量与上面选的样本一样

                d_loss = self.discriminator_model.train_on_batch([imgs, noise], [valid, fake, dummy])#训练判别器并计算损失，反向传播更新权重参数

            g_loss = self.generator_model.train_on_batch(noise, valid)#训练生成器并计算损失，反向传播更新权重参数

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], -100 * d_loss[1], g_loss))
            self.losslog.append([d_loss[0], g_loss])                                #存储损失值

            if epoch % self.sample_interval == 0:

                self.generator.save_weights('generator', overwrite=True)
                self.discriminator.save_weights('discriminator', overwrite=True)#每个self.sample_interval保存一次生成器和判别器的权重

    #生成少数类样本及选择合格少数类样本
        self.generator.load_weights('generator')
        k = m-q
        noise = np.random.normal(0, 1, (k, self.latent_dim))  # 选取cha个服从正态分布的噪声
        gen_imgs = self.generator.predict(noise)
        data = np.squeeze(gen_imgs)
        data = data.reshape(k, -1)
        data = data[:, 0:len(dataframe1.columns)].astype(np.float32)

        print("这一轮生成的少数类数量"+str(data.shape[0]))

        predicted1 = model[0].predict(data)
        predicted1.resize(len(predicted1), 1)

        predicted2 = model[1].predict(data)
        predicted2.resize(len(predicted2), 1)

        predicted3 = model[2].predict(data)
        predicted3.resize(len(predicted3), 1)

        predicted4 = model[3].predict(data)
        predicted4.resize(len(predicted4), 1)

        predicted5 = model[4].predict(data)
        predicted5.resize(len(predicted5), 1)

        predicted6 = model[5].predict(data)
        predicted6.resize(len(predicted6), 1)

        predicted7 = model[6].predict(data)
        predicted7.resize(len(predicted7), 1)

        predicted8 = model[7].predict(data)
        predicted8.resize(len(predicted8), 1)

        predicted9 = model[8].predict(data)
        predicted9.resize(len(predicted9), 1)

        predicted = np.concatenate((predicted1, predicted2, predicted3, predicted4, predicted5, predicted6, predicted7,
                                    predicted8, predicted9), axis=1)

        predicted = predicted.astype(int)
        i = 0
        z = 0
        u=m-w
        idnx = {}
        for c in range(k):
            c0 = 0.0
            c1 = 0.0
            for b in range(9):
                if predicted[c][b] == 1:
                    c1 += a[b]
                else:
                    c0 += a[b]
            if c1 > c0:
                idnx[i] = z
                i = i + 1
                u=u-1
            z = z + 1
            if u==0:
                break
        p = np.ones((i, data.shape[1]))
        print("这一轮合格的少数类数量"+str(p.shape[0]))

        for idx, id in idnx.items():
            p[idx] = data[id]

        p1 = pd.DataFrame(p)  # 转换为DataFrame 对象
        p1.to_csv('min.csv', header=False, index=False, mode='a')  # 保存为csv文件
        p1.to_csv('wgan-gp' + str(j) + '.csv', header=False, index=False, mode='a')  # 保存为csv文件
        w=w+i

        return w
if __name__ == '__main__':
    for i in range(1,4):
        epochs =5000
        batch_size = 32
        sample_interval = 1000
        wgan = WGANGP(epochs, batch_size)
        w,m,q=wgan.data_init(i)

        while(w<m):

            w=wgan.train(i,q)
            print("少数类总数量"+str(w))
        if os.path.exists('D:/pythonProject2/min.csv'):
                # 文件存在，执行删除操作
            os.remove('D:/pythonProject2/min.csv')
            print('文件 min.csv 删除成功！')
        else:
            # 文件不存在，输出提示信息
            print('文件 min.csv 不存在！无法删除。')
        print("第" + str(i) + "折数据集过采样结束")
