import copy
import csv
import time
import os
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch import optim
from torch import tensor
from sklearn.metrics import f1_score, auc, roc_auc_score, confusion_matrix, accuracy_score, roc_curve
from Models import Mnist_2NN, Mnist_CNN, MLP, LR, tabular_2NN, SVM, MLPclassify,DNN
from clients import ClientsGroup, client
from getData import GetDataSet, Get_Dataset_name
from compute_metrics import *
from aggregate_models import *


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=20, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=16, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='DNN', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.1, help="learning rate, \
                    use value from origin paper as default")   # 可以尝试在训练中不断减小lr，每 x 轮减少 y
parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")  # （通信的）模型验证频率
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')  # 全局模型保存频率（通信）
parser.add_argument('-ncomm', '--num_comm', type=int, default=150, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')

parser.add_argument('-resampling_method', '--ReS_method', type=str, default='NO', help='name of resampling method')
parser.add_argument('-isUndS', '--Undersampling', type=int, default=0, help='whether use a undersampling method in client data')
parser.add_argument('-UnS_name', '--undersampling_method', type=str, default='tl', help='the name of the undersampling method used')
parser.add_argument('-isOverS', '--Oversampling', type=int, default=1, help='whether use a oversampling method in client data')
parser.add_argument('-OvS_name', '--oversampling_method', type=str, default='SMOTE', help='the name of the oversampling method used')

# parser.add_argument('-dirichlet_alpha', '--alpha', type=float, default=1.0, help='dirichlet alpha')
# parser.add_argument('-dirichlet_seed', '--seed', type=int, default=42, help='dirichlet seed')
# parser.add_argument('-dataset_name', '--data_name', type=str, default='UNSWNB15', help='dataset name')
# parser.add_argument('-weighted_local_p', '--weighted', type=str, default='FA_R', help='FA-fedavg, GW-Gini-weighted')
parser.add_argument('-n_KF_Split', '--n_kf', type=int, default=10, help='10-kf or 5-kf')

def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__ == "__main__":
    args = parser.parse_args()
    args = args.__dict__

    # test_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # dataset_names = ['creditcard']
    # dataset_names = ['FraudDetection']
    # dataset_names = ['bean']
    # dataset_names = ['htru']
    # dataset_names = ['park']
    dataset_names = ['ieee']


    for dataset_namei in range(len(dataset_names)):
        args['data_name'] = dataset_names[dataset_namei]

        weighted_type = ['FedAvg']
        # weighted_type = ['Fedthree']
        # weighted_type = ['Fedthree_GINI']
        # weighted_type = ['Fedthree_inform']
        # weighted_type = ['Fed_inform']
        # weighted_type = ['Fedtwo_inform']
        for w_i in range(len(weighted_type)):
            args['weighted'] = weighted_type[w_i]

            if args['data_name'] in ['CM1']:
                args['num_of_clients'] = 5
                args['cfraction'] = 1
                args['model_name'] = 'MLP(15)'
                # args['model_name'] = 'tabular_2NN'
                # args['model_name'] = 'SVM'
                # args['model_name'] = 'LR'
                # args['model_name'] = 'MLPClassifly'
                # args['model_name'] = 'Mnist_2nn'
                args['alpha'] = 2.0
                args['seed'] = 10
            elif args['data_name'] in ['creditcard']:
                args['num_of_clients'] = 20
                args['cfraction'] = 0.5
                args['model_name'] = 'DNN'
                args['alpha'] = 2.0
                args['seed'] = 10
            elif args['data_name'] in ['ieee']:
                args['num_of_clients'] = 20
                args['cfraction'] = 0.5
                args['model_name'] = 'mlp'
                args['alpha'] = 2.0
                args['seed'] = 10
            else:
                exit('Error: unrecognized dataset')

            for i_kf in range(1):  # n 折交叉验证 args['n_kf']
                model = None
                if args['data_name'] in ['CM1']:
                    model = MLP(input_size=21, net=10)
                    # model = tabular_2NN()      #注意换数据集的时候该model的输入
                    # model = SVM(input_size=21,num_classes=2)
                    # model = LR(num_feature=21,output_size=2)
                    # model = MLPclassify()
                    # model = Mnist_2NN()
                elif args['data_name'] in ['Softwares_KC1']:
                    model = MLP(input_size=21, net=10)
                elif args['data_name'] in ['creditcard']:
                    # model = MLP(input_size=29, net=30)
                    model = DNN(input_size=29,output_size=2)
                elif args['data_name'] in ['ieee']:
                    # model = Mnist_CNN()
                    model = MLP(input_size=220, net=30)
                    # model = DNN(input_size=21,output_size=2)
                else:
                    exit('Error: unrecognized model')

                if torch.cuda.device_count() > 1:
                    print("Let's use", torch.cuda.device_count(), "GPUs!")
                    model = torch.nn.DataParallel(model)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)

                # 损失函数：交叉熵损失函数
                loss_func = F.cross_entropy

                opti = optim.SGD(model.parameters(), lr=args['learning_rate'], momentum=0.9,nesterov=True)  #, weight_decay=0.001)
                # opti = optim.Adam(model.parameters(), lr=args['learning_rate'], betas=(0.9, 0.99))
                # 数据集处理-------划分到 N 个 clients
                # 初始化时，载入 training 数据集
                myClients = ClientsGroup(args['data_name'], args['n_kf'], i_kf, args['IID'], args['Undersampling'], args['undersampling_method'],
                                         args['Oversampling'], args['oversampling_method'], args['num_of_clients'], args['alpha'], args['seed'], dev)
                testDataLoader = myClients.test_data_loader
                testset_labels = myClients.testset_label

                num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

                global_parameters = {}
                for key, var in model.state_dict().items():
                    global_parameters[key] = var.clone()

                data_file_path = 'D:/Project-python/A_Results_{}_S/{}/Test_{}_W({})_(C{}_D{}_S{}_{})'.format(
                    args['n_kf'],
                    args['data_name'],
                    args['ReS_method'],
                    args['weighted'],
                    args['num_of_clients'],
                    args['alpha'],
                    args['seed'],
                    args['model_name'])
                os.makedirs(data_file_path, exist_ok=True)

                results_file = 'KF_{}.csv'.format(i_kf)
                data_file = os.path.join(data_file_path, results_file)

                with open(data_file, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['global epoch', 'train_loss', 'test_loss', 'Training Time(s)', 'accuracy',
                                     'F1_score', 'G_mean', 'AUC'])  # 表头

                start_time = time.time()  # 计算每次通信的时间
                train_loss, train_accuracy = [], []
                test_loss = 0.0
                accuracy = 0
                F1_score = 0.0
                G_mean = 0.0
                AUC = 0.0
                np.random.seed(50)  # 设置随机种子为固定的整数，例如123

                # num_comm 表示通信次数
                for i in range(args['num_comm']):
                    local_losses = []
                    print("communicate round {}".format(i+1))

                    # 随机选择一部分client，全部选中会增大通信量，并且实验效果可能会不好
                    # client_in_comm表示每次通讯中随机选择的client
                    order = np.random.permutation(args['num_of_clients'])  # 随机排列函数,就是将输入的数据进行随机排列
                    clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]  # 是否每一轮通信都重新选择参与的客户端？？

                    sum_parameters = None
                    list_nums_local_data = []
                    list_nums_minclass_data = []  #记录每个客户端的少数类数量
                    list_alpk = []    #使用熵记录每个客户端的不平衡度
                    list_GINI = []    #使用GINI记录每个客户端的不平衡度
                    list_inform = []  #使用inform记录每个客户端的少数类价值
                    list_dicts_local_params = []
                    # 每个client基于当前的模型参数和自己的数据，训练和更新模型，返回每个client更新后的参数
                    for client in tqdm(clients_in_comm):
                        list_nums_local_data.append(copy.deepcopy(myClients.clients_set[client].Size_dataset))
                        list_nums_minclass_data.append(copy.deepcopy(myClients.clients_set[client].num_minclass))
                        list_alpk.append(copy.deepcopy(myClients.clients_set[client].alpk))
                        list_GINI.append(copy.deepcopy(myClients.clients_set[client].GINI))
                        # list_inform.append(copy.deepcopy(myClients.clients_set[client].inform))
                        # ------training------获取当前client训练得到的参数
                        local_parameters, loss = myClients.clients_set[client].localUpdate(args['epoch'],
                                                                                           args['batchsize'], model,
                                                                                           loss_func, opti,
                                                                                           global_parameters)  # 表示client端的训练函数
                        # for k,v in local_parameters.items():#差分隐私，加噪声
                        #     # noise = torch.tensor(np.random.laplace(0,10,local_parameters.shape)).to(dev)     #拉普拉斯机制实现差分隐私
                        #     noise = torch.cuda.FloatTensor(v.shape).normal_(0,10)               #高斯机制实现差分隐私
                        #     v.add_(noise)
                        local_losses.append(copy.deepcopy(loss))
                        list_dicts_local_params.append(copy.deepcopy(local_parameters))

                        # model.load_state_dict(local_parameters, strict=True)
                        # batch_loss = []
                        # num = 0
                        # preds_sum = tensor([]).to(dev)
                        # labels_sum = tensor([]).to(dev)
                        # # 载入测试集
                        # for data, label in testDataLoader:  # data 是 一个batchsize 的样本吗？？
                        #     data, label = data.to(dev), label.to(dev)
                        #     preds = model(data.float())
                        #     # print(preds.shape)
                        #     loss = loss_func(preds, label.long())
                        #     batch_loss.append(loss.item())
                        #     preds = torch.argmax(preds, dim=1)
                        #     # print(preds.shape)
                        #     preds_sum = torch.cat([preds_sum, preds], dim=0)
                        #     labels_sum = torch.cat([labels_sum, label], dim=0)
                        # # print(preds_sum.shape)
                        # # print(labels_sum.shape)
                        # test_loss = sum(batch_loss) / len(batch_loss)
                        # accuracy, F1_score, G_mean, AUC = metrics_compute(preds_sum, labels_sum)
                        #
                        # print('---F1_score: {}'.format(F1_score))
                        # print('---G_means: {}'.format(G_mean))
                        # print('---AUC: {}'.format(AUC))


                    # print(list_nums_minclass_data)
                    # print(list_nums_local_data)
                    # print(list_alpk)
                    if args['weighted'] == 'FedAvg':
                        global_parameters = aggrated_FedAvg(list_dicts_local_params, list_nums_local_data)
                    elif args['weighted'] == 'Fedthree':
                        global_parameters = aggrated_Fedthree(list_dicts_local_params, list_nums_local_data, list_nums_minclass_data, list_alpk)
                    elif args['weighted'] == 'Fedthree_GINI':
                        global_parameters = aggrated_Fedthree_GINI(list_dicts_local_params, list_nums_local_data, list_nums_minclass_data, list_GINI)
                    elif args['weighted'] == 'Fedthree_inform':
                        global_parameters = aggrated_Fedthree_inform(list_dicts_local_params, list_nums_local_data, list_inform, list_alpk)
                    elif args['weighted'] == 'Fed_inform':
                        global_parameters = aggrated_Fed_inform(list_dicts_local_params, list_inform)
                    elif args['weighted'] == 'Fedtwo_inform':
                        global_parameters = aggrated_Fedtwo_inform(list_dicts_local_params, list_nums_local_data, list_inform)
                    else:
                        exit('Error: unrecognized weighted type')

                    loss_avg = sum(local_losses) / len(local_losses)
                    train_loss.append(loss_avg)

                    # --------after a communication, to testing on the model in server for showing the performance-------------
                    with torch.no_grad():
                        # 加载server在最后得到的模型参数
                        # if (i + 1) % args['val_freq'] == 0:
                        model.load_state_dict(global_parameters, strict=True)
                        batch_loss = []
                        num = 0
                        preds_sum = tensor([]).to(dev)
                        labels_sum = tensor([]).to(dev)
                        # 载入测试集
                        for data, label in testDataLoader:  # data 是 一个batchsize 的样本吗？？
                            data, label = data.to(dev), label.to(dev)
                            preds = model(data.float())
                            # print(preds.shape)
                            loss = loss_func(preds, label.long())
                            batch_loss.append(loss.item())
                            preds = torch.argmax(preds, dim=1)
                            # print(preds.shape)
                            preds_sum = torch.cat([preds_sum, preds], dim=0)
                            labels_sum = torch.cat([labels_sum, label], dim=0)
                        # print(preds_sum.shape)
                        # print(labels_sum.shape)
                        test_loss = sum(batch_loss) / len(batch_loss)
                        accuracy, F1_score, G_mean, AUC = metrics_compute(preds_sum, labels_sum)

                        # print('---F1_score: {}'.format(F1_score))
                        # print('---G_means: {}'.format(G_means))
                        # print('---AUC: {}'.format(AUC))

                    with open(data_file, 'a', encoding='utf-8', newline='') as f:
                        writer = csv.writer(f)
                        # binary
                        writer.writerow([i + 1, train_loss[-1], test_loss, (time.time() - start_time), accuracy, F1_score, G_mean, AUC])            # multi
                        # writer.writerow([i + 1, train_loss[-1], test_loss, accuracy_test, (time.time() - start_time),
                        #                  accuracy_sklearn, F1_score_macro, F1_score_micro, F1_score_weighted])


