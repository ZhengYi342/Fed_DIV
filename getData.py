import numpy as np
import gzip
import os
import platform
import pickle
import pandas as pd


def Get_Dataset_name(data_name):
    if data_name == 'CICIDS17':
        data_dir = r'D:/Project-python/DataSets/binary/normalized/CICIDS17_normalize.csv'

    elif data_name == 'NSL_KDD':
        data_dir = r'D:/Project-python/DataSets/binary/normalized/NSL_KDD_normalize.csv'

    elif data_name == 'CSE_CIC_IDS2018':
        data_dir = r'D:/Project-python/DataSets/binary/normalized/CSE-CIC-IDS2018_normalize.csv'

    elif data_name == 'CSE_CIC_IDS2018(0.5)':
        data_dir = r'D:/Project-python/DataSets/binary/normalized/CSE-CIC-IDS2018_normalize(0.5).csv'

    elif data_name == 'CSE_CIC_IDS2018(0.8)':
        data_dir = r'D:/Project-python/DataSets/binary/normalized/CSE-CIC-IDS2018_normalize(0.8).csv'

    elif data_name == 'CSE_CIC_IDS2018_1':
        data_dir = r'D:/Project-python/DataSets/binary/normalized/CSE_CIC_IDS2018_normalize_1.csv'

    elif data_name == 'CSE_CIC_IDS2018_2(0.2)':
        data_dir = r'D:/Project-python/DataSets/binary/normalized/CSE_CIC_IDS2018_normalize_2(0.2).csv'

    elif data_name == 'UNSWNB15':
        data_dir = r'D:/Project-python/DataSets/binary/normalized/UNSWNB15_normalize_updated.csv'

    elif data_name == 'KDDCUP1999':
        data_dir = r'D:/Project-python/DataSets/binary/normalized/KDDCup1999_normalize.csv'

    else:
        print('无名为 {} 的数据集'.format(data_name))

    dataset = pd.read_csv(data_dir, header=None)
    return dataset


class GetDataSet_KFold(object):
    def __init__(self, dataSetName, n_kf, i_KF):
        self.name = dataSetName
        self.train_datasets = None  # 包含属性和标签的训练数据集
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None

        self._index_in_train_epoch = 0

        data_dir_train = r'D:/Project-python/DataSets/{}_KF/{}/train_dataset_{}.csv'.format(n_kf, dataSetName, i_KF)
        data_dir_test = r'D:/Project-python/DataSets/{}_KF/{}/test_dataset_{}.csv'.format(n_kf, dataSetName, i_KF)
        train_dataset = pd.read_csv(data_dir_train, header=None)
        test_dataset = pd.read_csv(data_dir_test, header=None)

        self.train_datasets = train_dataset
        self.train_data = train_dataset.iloc[:, 0:train_dataset.shape[1] - 1]
        self.train_label = train_dataset.iloc[:, -1]

        self.test_data = test_dataset.iloc[:, 0:test_dataset.shape[1] - 1]
        self.test_label = test_dataset.iloc[:, -1]


class GetDataSet(object):
    def __init__(self, dataSetName, isIID):
        self.name = dataSetName
        self.train_datasets = None  # 包含属性和标签的训练数据集
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None

        self._index_in_train_epoch = 0

        if self.name == 'mnist':
            self.mnistDataSetConstruct(isIID)
        elif self.name == 'adult':
            self.adultDataSetConstruct()

        elif self.name == 'UNSW_NB15':
            data_dir_train = r'D:/Project-python/DataSets/multi/trian_test_data/UNSW_NB15_train_8.csv'
            data_dir_test = r'D:/Project-python/DataSets/multi/trian_test_data/UNSW_NB15_test_8.csv'
            self.DataSetConstruct(data_dir_train, data_dir_test)
        elif self.name == 'magic':
            data_dir_train = r'D:/Project-python/FedAvg-master - (2)/use_pytorch/data/magic/magic_train.csv'
            data_dir_test = r'D:/Project-python/FedAvg-master - (2)/use_pytorch/data/magic/magic_test.csv'
            self.DataSetConstruct(data_dir_train, data_dir_test)
        elif self.name == 'coil2000':
            data_dir_train = r'D:/Project-python/FedAvg-master - (2)/use_pytorch/data/coil2000/coil2000_train.csv'
            data_dir_test = r'D:/Project-python/FedAvg-master - (2)/use_pytorch/data/coil2000/coil2000_test.csv'
            self.DataSetConstruct(data_dir_train, data_dir_test)
        elif self.name == 'spambase':
            data_dir_train = r'D:/Project-python/FedAvg-master - (2)/use_pytorch/data/spambase/spambase_train.csv'
            data_dir_test = r'D:/Project-python/FedAvg-master - (2)/use_pytorch/data/spambase/spambase_test.csv'
            self.DataSetConstruct(data_dir_train, data_dir_test)
        elif self.name == 'phoneme':
            data_dir_train = r'D:/Project-python/FedAvg-master - (2)/use_pytorch/data/phoneme/phoneme_train.csv'
            data_dir_test = r'D:/Project-python/FedAvg-master - (2)/use_pytorch/data/phoneme/phoneme_test.csv'
            self.DataSetConstruct(data_dir_train, data_dir_test)
        elif self.name == 'CICIDS17':
            data_dir_train = r'D:/Project-python/DataSets/binary/train_test_data/CICIDS17_normalize_train.csv'
            data_dir_test = r'D:/Project-python/DataSets/binary/train_test_data/CICIDS17_normalize_test.csv'
            self.DataSetConstruct(data_dir_train, data_dir_test)
        elif self.name == 'CICIDS17(0.4)':
            data_dir_train = r'D:/Project-python/DataSets/binary/train_test_data/CICIDS17_normalize(0.4)_train.csv'
            data_dir_test = r'D:/Project-python/DataSets/binary/train_test_data/CICIDS17_normalize(0.4)_test.csv'
            self.DataSetConstruct(data_dir_train, data_dir_test)

        elif self.name == 'NSL_KDD':
            data_dir_train = r'D:/Project-python/DataSets/binary/train_test_data/NSL_KDD_normalize_train.csv'
            data_dir_test = r'D:/Project-python/DataSets/binary/train_test_data/NSL_KDD_normalize_test.csv'
            self.DataSetConstruct(data_dir_train, data_dir_test)

        elif self.name == 'CSE_CIC_IDS2018':
            data_dir_train = r'D:/Project-python/DataSets/binary/train_test_data/CSE-CIC-IDS2018_normalize_train.csv'
            data_dir_test = r'D:/Project-python/DataSets/binary/train_test_data/CSE-CIC-IDS2018_normalize_test.csv'
            self.DataSetConstruct(data_dir_train, data_dir_test)
        elif self.name == 'CSE_CIC_IDS2018(0.5)':
            data_dir_train = r'D:/Project-python/DataSets/binary/train_test_data/CSE-CIC-IDS2018_normalize(0.5)_train.csv'
            data_dir_test = r'D:/Project-python/DataSets/binary/train_test_data/CSE-CIC-IDS2018_normalize(0.5)_test.csv'
            self.DataSetConstruct(data_dir_train, data_dir_test)
        elif self.name == 'CSE_CIC_IDS2018(0.8)':
            data_dir_train = r'D:/Project-python/DataSets/binary/train_test_data/CSE-CIC-IDS2018_normalize(0.8)_train.csv'
            data_dir_test = r'D:/Project-python/DataSets/binary/train_test_data/CSE-CIC-IDS2018_normalize(0.8)_test.csv'
            self.DataSetConstruct(data_dir_train, data_dir_test)

        elif self.name == 'CSE_CIC_IDS2018_1':
            data_dir_train = r'D:/Project-python/DataSets/binary/train_test_data/CSE_CIC_IDS2018_1_train.csv'
            data_dir_test = r'D:/Project-python/DataSets/binary/train_test_data/CSE_CIC_IDS2018_1_test.csv'
            self.DataSetConstruct(data_dir_train, data_dir_test)

        elif self.name == 'CSE_CIC_IDS2018_2':
            data_dir_train = r'D:/Project-python/DataSets/binary/train_test_data/CSE_CIC_IDS2018_2_train.csv'
            data_dir_test = r'D:/Project-python/DataSets/binary/train_test_data/CSE_CIC_IDS2018_2_test.csv'
            self.DataSetConstruct(data_dir_train, data_dir_test)

        elif self.name == 'CSE_CIC_IDS2018_2(0.5)':
            data_dir_train = r'D:/Project-python/DataSets/binary/train_test_data/CSE_CIC_IDS2018_2(0.5)_train.csv'
            data_dir_test = r'D:/Project-python/DataSets/binary/train_test_data/CSE_CIC_IDS2018_2(0.5)_test.csv'
            self.DataSetConstruct(data_dir_train, data_dir_test)

        elif self.name == 'CSE_CIC_IDS2018_2(0.2)':
            data_dir_train = r'D:/Project-python/DataSets/binary/train_test_data/CSE_CIC_IDS2018_2(0.2)_train.csv'
            data_dir_test = r'D:/Project-python/DataSets/binary/train_test_data/CSE_CIC_IDS2018_2(0.2)_test.csv'
            self.DataSetConstruct(data_dir_train, data_dir_test)

        elif self.name == 'UNSWNB15':
            data_dir_train = r'D:/Project-python/DataSets/binary/train_test_data/UNSWNB15_normalize_train.csv'
            data_dir_test = r'D:/Project-python/DataSets/binary/train_test_data/UNSWNB15_normalize_test.csv'
            self.DataSetConstruct(data_dir_train, data_dir_test)

        elif self.name == 'KDDCUP1999':
            data_dir_train = r'D:/Project-python/DataSets/binary/train_test_data/KDDCup1999_normalize_train.csv'
            data_dir_test = r'D:/Project-python/DataSets/binary/train_test_data/KDDCup1999_normalize_test.csv'
            self.DataSetConstruct(data_dir_train, data_dir_test)

        elif self.name == 'new_data':
            data_dir_train = r'D:/Project-python/DataSets/binary/naticusdroid+android+permissions+dataset/data_train.csv'
            data_dir_test = r'D:/Project-python/DataSets/binary/naticusdroid+android+permissions+dataset/data_test.csv'
            self.DataSetConstruct(data_dir_train, data_dir_test)
        else:
            pass

    def DataSetConstruct(self, data_dir_train, data_dir_test):
        train_dataset = pd.read_csv(data_dir_train, header=None)
        test_dataset = pd.read_csv(data_dir_test, header=None)

        self.train_datasets = train_dataset
        self.train_data = train_dataset.iloc[:, 0:train_dataset.shape[1] - 1]
        self.train_label = train_dataset.iloc[:, -1]

        self.test_data = test_dataset.iloc[:, 0:test_dataset.shape[1] - 1]
        self.test_label = test_dataset.iloc[:, -1]


    def adultDataSetConstruct(self):
        data_dir = r'.\data\adult'
        data_dir_train = '../data/adult/adult_train.csv'
        data_dir_test = '../data/adult/adult_test.csv'
        train_dataset = pd.read_csv(data_dir_train, header=None)
        test_dataset = pd.read_csv(data_dir_test, header=None)

        self.train_datasets = train_dataset
        self.train_data = train_dataset.iloc[:, 0:train_dataset.shape[1] - 1]
        self.train_label = train_dataset.iloc[:, -1]

        self.test_data = test_dataset.iloc[:, 0:test_dataset.shape[1] - 1]
        self.test_label = test_dataset.iloc[:, -1]

    def mnistDataSetConstruct(self, isIID):
        data_dir = r'.\data\MNIST'
        # data_dir = r'./data/MNIST'
        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
        train_images = extract_images(train_images_path)
        train_labels = extract_labels(train_labels_path)
        test_images = extract_images(test_images_path)
        test_labels = extract_labels(test_labels_path)

        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        assert train_images.shape[3] == 1
        assert test_images.shape[3] == 1
        train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2])
        test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])

        train_images = train_images.astype(np.float32)
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        if isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:
            labels = np.argmax(train_labels, axis=1)
            order = np.argsort(labels)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]



        self.test_data = test_images
        self.test_label = test_labels


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                    'Invalid magic number %d in MNIST label file: %s' %
                    (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return dense_to_one_hot(labels)

#
# if __name__=="__main__":
#     'test data set'
#     mnistDataSet = GetDataSet('mnist', True) # test NON-IID
#     if type(mnistDataSet.train_data) is np.ndarray and type(mnistDataSet.test_data) is np.ndarray and \
#             type(mnistDataSet.train_label) is np.ndarray and type(mnistDataSet.test_label) is np.ndarray:
#         print('the type of data is numpy ndarray')
#     else:
#         print('the type of data is not numpy ndarray')
#     print('the shape of the train data set is {}'.format(mnistDataSet.train_data.shape))
#     print('the shape of the test data set is {}'.format(mnistDataSet.test_data.shape))
#     print(mnistDataSet.train_label[0:100], mnistDataSet.train_label[11000:11100])

