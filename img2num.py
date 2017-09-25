from mnist import MNIST
from torch.autograd import Variable
import torch
import numpy as np

class Img2Num(object):
    def __init__(self):
        # Load MNIST
        mndata = MNIST('./python-mnist/data')
        self.train_data, self.train_label = mndata.load_training()
        # oneHot encoding
        label = []
        max_value = max(self.train_label)
        min_value = min(self.train_label)
        for l in self.train_label:
            label.append([1 if i==l else 0 for i in range(max_value-min_value+1)])
        self.train_label = label
        in_layer = len(self.train_data[0])
        out_layer = len(self.train_label[0])
        # Intialize NeuralNetwork
        self.train_data = torch.FloatTensor(self.train_data)
        self.train_label = torch.FloatTensor(self.train_label)
        self.model = torch.nn.Sequential(
          torch.nn.Linear(in_layer, in_layer/2),
          torch.nn.Sigmoid(),
          torch.nn.Linear(in_layer/2, out_layer),
          torch.nn.Sigmoid(),
        )
        self.loss_function = torch.nn.MSELoss()
        self.eta = 1e-2
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.eta)

    def forward(self, img):
        img = torch.FloatTensor(img)
        if len(img.size()) == 3:
            img = img.view(img.size()[0], img.size()[1]*img.size()[2])
        elif len(img.size()) == 2:
            img = img.view(img.size()[0]*img.size()[1])
        output = self.model(Variable(torch.FloatTensor(img))).data
        # return output
        if len(output.size())==2:
            return np.argmax(output.numpy(), 1)
        else:
            return np.argmax(output.numpy())

    def train(self):
        batch_size = 128
        # print(type(self).__name__, "Start training")
        current_index = 0
        num_train_data = self.train_data.size()[0]
        while current_index < num_train_data:
            td = Variable(self.train_data[current_index:current_index+batch_size])
            tl = Variable(self.train_label[current_index:current_index+batch_size])
            self.optimizer.zero_grad()
            pred_label = self.model(td)
            loss = self.loss_function(pred_label, tl)
            loss.backward()
            self.optimizer.step()
            current_index += td.size()[0]
        # print(type(self).__name__, "Finish training")
