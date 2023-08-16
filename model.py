import torch
from torch.nn import functional as F

class myConNet(torch.nn.Module):
    def __init__(self):
        super(myConNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(6, 12, kernel_size=5)

        self.fc1 = torch.nn.Linear(12 * 4 * 4, 120)
        self.fc2 = torch.nn.Linear(120, 60)

        self.out = torch.nn.Linear(60, 10)
    
    def forward(self, t):
        # 第一层卷积和池化处理
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        # 第二层卷积和池化处理
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        # 搭建全链接网络，第一层全连接
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)
        # 第二层全连接
        t = self.fc2(t)
        t = F.relu(t)
        # 第三层连接
        t = self.out(t)
        return t

if __name__ == '__main__':
    network = myConNet()
    # 指定设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    network.to(device)
    print(network)
