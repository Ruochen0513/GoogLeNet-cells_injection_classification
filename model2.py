import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        # 第一条分支：1x1卷积
        self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)

        # 第二条分支：1x1卷积 + 3x3卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),  # 1x1卷积
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)  # 3x3卷积，保持输入和输出尺寸一致
        )

        # 第三条分支：1x1卷积 + 5x5卷积
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),  # 1x1卷积
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)  # 5x5卷积，保持输入和输出尺寸一致
        )

        # 第四条分支：3x3最大池化 + 1x1卷积
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),  # 3x3最大池化，保持输入和输出尺寸一致
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)  # 1x1卷积
        )

    def forward(self, x):
        # 将输入分别传入四个分支
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        # 将四个分支的输出在通道维度上进行拼接
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(GoogLeNet, self).__init__()

        # 第一层卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 第二层卷积层
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 第三层Inception模块
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 第四层Inception模块
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 第五层Inception模块
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        # 平均池化层，Dropout层，全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # 通过第一层卷积和池化
        x = self.maxpool1(F.relu(self.conv1(x)))

        # 通过第二层卷积和池化
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.maxpool2(x)

        # 通过第三层Inception模块
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        # 通过第四层Inception模块
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        # 通过第五层Inception模块
        x = self.inception5a(x)
        x = self.inception5b(x)

        # 通过平均池化、Dropout和全连接层
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.dropout(x)
        x = self.fc(x)
        return x
