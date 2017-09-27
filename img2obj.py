from torch.autograd import Variable
from torch import nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class Img2Obj(nn.Module):
    def __init__(self):
        self.num_classes = 100
        super(Img2Obj, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Conv2d(16, 120, 5)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)

    def forward(self, x):
        x = x.float()
        if len(x.size()) == 3:
            (C, H, W) = x.data.size()
            img = img.view(1, C, H, W)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.fc1(x)
        x = x.squeeze(2).squeeze(2)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        return x

    def view(self, img):
        x = img.float()
        if len(x.size()) == 3:
            (C, H, W) = x.size()
            x = x.view(1, C, H, W)
        x_pred = self.forward(Variable(x))
        x_pred = np.argmax(x_pred.data.numpy(), 1)
        plt.imshow(img.transpose(0, 2).transpose(0, 1).numpy())
        plt.title("Object Class: {}".format(x_pred))
        plt.show()


    def train(self):
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=1e-2)
        # Load CIFAR100
        root = 'torchvision/CIFAR-100/'
        download = False
        trans = transforms.Compose([transforms.ToTensor()])
        train_set = dset.CIFAR100(root=root, train=True, transform=trans, download=download)
        batch_size = 256
        train_loader = torch.utils.data.DataLoader(
                         dataset=train_set,
                         batch_size=batch_size,
                         shuffle=True)
        epoch = 1
        for i in range(epoch):
            # training
            batch_idx = 0
            for batch_idx, (x, target) in enumerate(train_loader):
                self.optimizer.zero_grad()

                x, target = Variable(x), Variable(Img2Obj.oneHot(target, self.num_classes))
                x_pred = self.forward(x)
                loss = self.loss_function(x_pred, target)
                loss.backward()
                self.optimizer.step()
                if (batch_idx+1)% 10 == 0:
                    # print '==>>> batch index: {}, train loss: {:.6f}'.format(batch_idx, loss.data[0])
                    print '==>>> batch index: {}'.format(batch_idx+1)
            print '==>>> batch index: {}/{}'.format(batch_idx+1, len(train_loader))

    @staticmethod
    def oneHot(target, num_classes):
        # oneHot encoding
        label = []
        for l in target:
                label.append([1 if i==l else 0 for i in range(num_classes)])
        return torch.FloatTensor(label)

    def cam(self, *idx):
        pass
