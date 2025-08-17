import torch
import torch.nn as nn
import torch.nn.functional as F


class tabular_2NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(21, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 2)

    def forward(self, inputs):
        tensor = F.relu(self.fc1(inputs))
        tensor = F.relu(self.fc2(tensor))
        tensor = self.fc3(tensor)
        return tensor


class SVM(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)


class LR(nn.Module):
    # Logistics Regression
    def __init__(self, num_feature, output_size):
        super(LR, self).__init__()

        self.num_feature = num_feature
        self.output_size = output_size
        self.linear = nn.Linear(self.num_feature, self.output_size)
        self.sigmoid = nn.Sigmoid()
        self.model = nn.Sequential(self.linear, self.sigmoid)

    def forward(self, x):
        # x = x.view(-1, 10)
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, input_size, net):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(input_size, net)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(net, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)

class MLPclassify(nn.Module):
    def __init__(self):
        super(MLPclassify, self).__init__()
        # 定义第一个隐藏层
        self.hidden1 = nn.Sequential(
            nn.Linear(
                in_features=21,  # 第一个隐藏层的输入，数据的特征数
                out_features=30,  # 第一个隐藏层的输出，神经元的数量
                bias=True
            ),
            nn.ReLU()
        )
        # 定义第二个隐藏层
        self.hidden2 = nn.Sequential(
            nn.Linear(30, 10),
            nn.ReLU()
        )
        # 分类层
        self.classify = nn.Sequential(
            nn.Linear(10, 2),
            nn.Sigmoid()
        )

    # 定义网络的前向传播
    def forward(self, x):
        fc1 = self.hidden1(x)
        fc2 = self.hidden2(fc1)
        output = self.classify(fc2)
        # 输出为两个隐藏层和两个输入层
        return output

class Mnist_2NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(21, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, inputs):
        tensor = F.relu(self.fc1(inputs))
        tensor = F.relu(self.fc2(tensor))
        tensor = self.fc3(tensor)
        return tensor


class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(1* 16 * 64, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 16, 16)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 1 * 16 * 64)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor


class DNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DNN, self).__init__()

        self.e1 = nn.Linear(input_size, 2 * input_size)

        self.e2 = nn.Linear(2 * input_size, 3 * input_size)

        self.e3 = nn.Linear(3 * input_size, output_size)


    def forward(self, z):
        x = self.e1(z)
        x = F.relu(x)

        x = self.e2(x)
        x = F.relu(x)

        x = self.e3(x)
        prob = F.softmax(x, dim=1)
        return prob

    def cv(self, z):
        x = self.e1(z)
        x = F.relu(x)
        return x
class CNN(nn.Module):
    def __init__(self, in_features=1, num_classes=2, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out