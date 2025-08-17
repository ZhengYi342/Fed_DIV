import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getData import GetDataSet, GetDataSet_KFold
from sample_dirichlet import Partitioner
import pandas as pd
from ReSampling import *
import numpy as np
import os
import csv
from math import log


class client(object):
    def __init__(self, trainDataSet, num_minclass, alpk, GINI, dev):#inform
    # def __init__(self, trainDataSet, num_minclass, alpk, GINI, inform, dev):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None

        self.Size_dataset = len(trainDataSet)
        self.num_minclass = num_minclass
        self.alpk = alpk
        self.GINI = GINI
        # self.inform = inform

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters):
        '''
          :param localEpoch: 当前Client的迭代次数
          :param localBatchSize: 当前Client的batchsize大小
          :param Net: Server共享的模型
          :param lossFun: 损失函数
          :param opti: 优化函数
          :param global_parameters: 当前通信中最新全局参数
          :return: 返回当前Client基于自己的数据训练得到的新的模型参数
        '''
        # 加载当前通信中最新全局参数
        Net.load_state_dict(global_parameters, strict=True)
        # 加载client自有数据集
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)  # 这里对训练集划分了num/batch_size 个 batch？？

        epoch_acc = []
        epoch_loss = []
        # 本地迭代次数
        for epoch in range(localEpoch):
            batch_loss = []
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data.float())
                loss = lossFun(preds, label.long())
                loss.backward()

                # for name, parms in Net.named_parameters():  # 查看是否有梯度回传,查看代码如下:
                #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data),
                #           ' -->grad_value:', torch.mean(parms.grad))

                opti.step()
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opti, 'min')
                scheduler.step(loss)  # 增加根据loss的值变换调整学习率

                batch_loss.append(loss.item())
                opti.zero_grad()

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        # 返回当前client基于自己数据集训练得到的新的模型参数
        return Net.state_dict(), sum(epoch_loss)/len(epoch_loss)

    def local_val(self):
        pass


def eucliDist(A,B):
    return np.sqrt(sum(np.power((A - B), 2)))

class ClientsGroup(object):
    def __init__(self, dataSetName, n_kf, i_kf, isIID, isUnS, UnS_name, isOvS, OvS_name, numOfClients, dirichlet_alpha, seed_v, dev):
        self.data_set_name = dataSetName
        self.n_kf = n_kf
        self.i_KF = i_kf
        self.is_iid = isIID
        self.is_UnS = isUnS
        self.Uns_name = UnS_name
        self.isOvS = isOvS
        self.Ovs_name = OvS_name
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}
        self.dirichlet = dirichlet_alpha
        self.seed_value = seed_v

        self.test_data_loader = None   # 在函数 dataSetBalanceAllocation() 中有 test_data_loader 的赋值 函数内第 4 条语句
        self.testset_label = None

        # self.dataSetBalanceAllocation()  # 数据集
        self.dataSet_DirichletAllocation()
        # self.data_temporary_save()  # 直接读取处理好的客户端数据集


    def dataSet_DirichletAllocation(self):
        # DataSet = GetDataSet(self.data_set_name, self.is_iid)
        DataSet = GetDataSet_KFold(self.data_set_name, self.n_kf, self.i_KF)

        test_data = torch.tensor(DataSet.test_data.values)
        test_label = torch.tensor(DataSet.test_label.values)
        self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=512, shuffle=False)
        self.testset_label = test_label
        train_dataset = DataSet.train_datasets

        non_iid_partitioner = Partitioner(train_dataset, self.num_of_clients, self.dirichlet, self.seed_value,show_img=True)
        user_groups = non_iid_partitioner.partition()
        print("狄雷克雷划分完毕")
        # 根据获得的数据索引，将样本 划分给客户端
        for i in range(self.num_of_clients):  # 为clients 划分 local数据
            client_index = list(user_groups[i])
            client_data = train_dataset.loc[client_index]  # client_data: dataframe

            local_data = client_data.iloc[:, 0:client_data.shape[1] - 1].values  #np类型
            local_label = client_data.iloc[:, -1].values.astype(int)

            # num_class_0, num_class_1, total_num_samples, minclass, num_majclass, num_minclass
            num_minclass, alpk, GINI = self.imbalance_data_info(client_data)
            # num_minclass, alpk, GINI, inform = self.imbalance_data_info(client_data)

            data = pd.DataFrame(local_data)
            label = pd.DataFrame(local_label)
            data.to_csv("特征存储路径", header=False, index=False, mode='a')
            label.to_csv("标签存储路径", header=False,index=False, mode='a')#存储每个客户端分配的数据，以便后续过采样
            print(str(self.i_KF)+"_"+str(i)+"_"+"分配结束")
            # resampling-undersampling
            if self.is_UnS:
                U_local_data, U_local_label = UnderSampling_methods(local_data, local_label, self.Uns_name)
                someone = client(TensorDataset(torch.tensor(U_local_data), torch.tensor(U_local_label)), num_minclass, alpk, GINI, self.dev) #inform

            # resampling-oversampling
            elif self.isOvS:
                O_local_data, O_local_label = OverSampling_methods(local_data, local_label, self.Ovs_name)
                someone = client(TensorDataset(torch.tensor(O_local_data), torch.tensor(O_local_label)), num_minclass, alpk, GINI, self.dev)

            else:
                someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), num_minclass, alpk, GINI, self.dev)
                # someone是 client类的实例化对象，client(self,trainDataset,dev)

            self.clients_set['client{}'.format(i)] = someone  # self.clients_set（dict），存储内容为 class object
    #直接读取处理好的客户端数据集,用这个
    def data_temporary_save(self):
        DataSet = GetDataSet_KFold(self.data_set_name, self.n_kf, self.i_KF)

        test_data = torch.tensor(DataSet.test_data.values)
        test_label = torch.tensor(DataSet.test_label.values)
        self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=512, shuffle=False)
        self.testset_label = test_label
        train_dataset = DataSet.train_datasets

        non_iid_partitioner = Partitioner(train_dataset, self.num_of_clients, self.dirichlet, self.seed_value,
                                          show_img=True)
        user_groups = non_iid_partitioner.partition()

        for i in range(self.num_of_clients):  # 为clients 划分 local数据
            client_index = list(user_groups[i])
            client_d = train_dataset.loc[client_index]  # client_data: dataframe
            client_data = pd.read_csv("已经过采样完毕的数据路径", header=None)
            local_data = client_data.iloc[:, 0:client_data.shape[1] - 1].values  # np类型
            local_label = client_data.iloc[:, -1].values.astype(int)

            # num_class_0, num_class_1, total_num_samples, minclass, num_majclass, num_minclass
            num_minclass, alpk, GINI, inform = self.imbalance_data_info(client_d)

            someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), num_minclass, alpk, GINI, inform, self.dev)
            self.clients_set['client{}'.format(i)] = someone  # self.clients_set（dict），存储内容为 class object
            print("客户端"+str(i)+"分配与计算完毕")

    def imbalance_data_info(self, client_data):
        shannon_ent = 0.0
        num_class_0 = (client_data.iloc[:, -1] == 0).sum()
        num_class_1 = (client_data.iloc[:, -1] == 1).sum()
        total_num_samples = client_data.shape[0]
        if num_class_0 > num_class_1:
            majclass = 0
            minclass = 1
            num_majclass = num_class_0
            num_minclass = num_class_1
        else:
            majclass = 1
            minclass = 0
            num_majclass = num_class_1
            num_minclass = num_class_0
        k =  len(client_data.columns)-1
        dfmin = client_data[client_data[k] == 1]  #dataframe类型,少数类
        data_all_class = client_data.values    # 所有数据，带类别
        data_min_class = dfmin.values          # 少数类数据，带类别
        data_all = data_all_class[:, 0:k].astype(np.float32)  # 所有数据
        data_min = data_min_class[:, 0:k].astype(np.float32)  # 少数类数据
        dist = np.zeros((data_all.shape[0] - 1, 2))  # 存储距离的数组
        IX = [0.0] * data_min.shape[0]  # 存储信息值的列表
        N_k = 5  # 所指定的近邻数
        # for g in range(data_min.shape[0]):
        #     count = 0
        #     for h in range(data_all.shape[0]):
        #         if g != h:
        #             dist_temp = eucliDist(data_min[g], data_all[h])
        #             dist[count][0] = dist_temp
        #             dist[count][1] = data_all_class[h][k]
        #             count += 1
        #     dist_sort = dist[np.argsort(dist[:, 0])]
        #     Dall = 0.0
        #     Dmaj = 0.0
        #     Nmaj = 0
        #     for h in range(N_k):  # k个近邻
        #         if dist_sort[h][1] == 0:
        #             Nmaj += 1
        #             Dmaj += dist_sort[h][0]
        #         Dall += dist_sort[h][0]
        #     if Dmaj == 0.0 or Dall == 0.0:
        #         DX = 0.0
        #     else:
        #         DX = Dmaj / Dall
        #     CX = Nmaj / N_k
        #     IX[g] = DX + CX
        # sum = 0.0
        # for g in range(len(IX)):
        #     sum += IX[g]
        #
        # inform = sum
        alpk = -((num_minclass/total_num_samples)*log(num_minclass/total_num_samples))-((num_majclass/total_num_samples)*log(num_majclass/total_num_samples))
        GINI = 1 - pow(num_minclass/total_num_samples,2)-pow(num_majclass/total_num_samples,2)
        imbalance_rate = num_majclass / num_minclass  # 少数类的占比
        IR = num_minclass / total_num_samples

        # return num_minclass, alpk, GINI
        return num_minclass, alpk, GINI          #inform ，num_class_0, num_class_1, total_num_samples, minclass, num_majclass, num_minclass, imbalance_rate, IR


#     def dataSetBalanceAllocation(self):
#         mnistDataSet = GetDataSet(self.data_set_name, self.is_iid)  # 数据集
#
#         test_data = torch.tensor(mnistDataSet.test_data)
#         test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)
#         self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)
#
#         train_data = mnistDataSet.train_data
#         train_label = mnistDataSet.train_label
#
#         shard_size = mnistDataSet.train_data_size // self.num_of_clients // 2
#         shards_id = np.random.permutation(mnistDataSet.train_data_size // shard_size)
#         for i in range(self.num_of_clients):  # 为 100 clients 划分 local数据
#             shards_id1 = shards_id[i * 2]
#             shards_id2 = shards_id[i * 2 + 1]
#             data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
#             data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
#             label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
#             label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
#             local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
#             local_label = np.argmax(local_label, axis=1)
#             someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
#             self.clients_set['client{}'.format(i)] = someone
#
# if __name__=="__main__":
#     MyClients = ClientsGroup('mnist', True, 100, 1)
#     print(MyClients.clients_set['client10'].train_ds[0:100])
#     print(MyClients.clients_set['client11'].train_ds[400:500])


