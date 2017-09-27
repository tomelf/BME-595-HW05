from img2num import Img2Num
from img2obj import Img2Obj
from nn_img2num import NnImg2Num
from torch.autograd import Variable
from torch import nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time

def main():
    # PartA
    testPartA = False
    testPartB = True

    if testPartA:
        print("Load MNIST")
        # Load MNIST
        root = 'torchvision/mnist/'
        download = True
        trans = transforms.Compose([transforms.ToTensor()])
        train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
        test_set = dset.MNIST(root=root, train=False, transform=trans)

        train_loader = torch.utils.data.DataLoader(
                         dataset=train_set,
                         batch_size=len(train_set),
                         shuffle=True)
        test_loader = torch.utils.data.DataLoader(
                        dataset=test_set,
                        batch_size=len(test_set),
                        shuffle=True)

        models = [NnImg2Num, Img2Num]

        for model in models:
            print("Start {0} model".format(type(model()).__name__))
            training_errors = []
            testing_errors = []
            training_time = []

            epoch = 20
            md = model()

            print("== Start training for {0:d} epochs".format(epoch))
            for i in range(epoch):
                # training
                start_time = time.time()
                md.train()
                if i == 0:
                    training_time.append(time.time()-start_time)
                else:
                    training_time.append(time.time()-start_time+training_time[-1])
                print("-- Finish epoch {0:d}".format(i+1))
                print '==>>> epoch: {}, training time: {:.6f}'.format(i+1, training_time[-1])
                # training error
                print '==>>> verify training error'
                for batch_idx, (x, target) in enumerate(train_loader):
                    x_pred = md.forward(Variable(x))
                    x_pred = np.argmax(x_pred.data.numpy(), 1)
                    accu = accuracy_score(target.numpy(), x_pred)
                    if batch_idx == len(test_loader)-1:
                        print '==>>> epoch: {}, training error: {:.6f}'.format(i+1, 1-accu)
                        training_errors.append(1-accu)
                # testing error
                print '==>>> verify testing error'
                for batch_idx, (x, target) in enumerate(test_loader):
                    x_pred = md.forward(Variable(x))
                    x_pred = np.argmax(x_pred.data.numpy(), 1)
                    accu = accuracy_score(target.numpy(), x_pred)
                    if batch_idx == len(test_loader)-1:
                        print '==>>> epoch: {}, testing error: {:.6f}'.format(i+1, 1-accu)
                        testing_errors.append(1-accu)

                plt.title(type(model()).__name__)
                plt.xlabel("epochs")
                plt.ylabel("error")
                plt.plot(range(i+1), training_errors, "bo", range(i+1), training_errors, "b--", label="training_error")
                plt.plot(range(i+1), testing_errors, "ro", range(i+1), testing_errors, "r--", label="testing_error")
                plt.legend(loc='upper right')
                plt.xticks(range(i+1))
                plt.savefig("{0}_error_ep-{1:d}.png".format(type(model()).__name__, i+1))
                plt.clf()

                plt.title(type(model()).__name__)
                plt.xlabel("epochs")
                plt.ylabel("seconds")
                plt.plot(range(i+1), training_time, "bo", range(i+1), training_time, "b--", label="training_time")
                plt.legend(loc='upper left')
                plt.xticks(range(i+1))
                plt.savefig("{0}_speed_ep-{1:d}.png".format(type(model()).__name__, i))
                plt.clf()
            print("Done!")

    if testPartB:
        print("Load CIFAR-100")
        # Load CIFAR-100
        root = 'torchvision/CIFAR-100/'
        download = False
        trans = transforms.Compose([transforms.ToTensor()])
        train_set = dset.CIFAR100(root=root, train=True, transform=trans, download=download)
        test_set = dset.CIFAR100(root=root, train=False, transform=trans)

        train_loader = torch.utils.data.DataLoader(
                         dataset=train_set,
                         batch_size=len(train_set),
                         shuffle=True)
        test_loader = torch.utils.data.DataLoader(
                        dataset=test_set,
                        batch_size=len(test_set),
                        shuffle=True)

        models = [Img2Obj]

        for model in models:
            print("Start {0} model".format(type(model()).__name__))
            training_errors = []
            testing_errors = []
            training_time = []

            epoch = 20
            md = model()

            print("== Start training for {0:d} epochs".format(epoch))
            for i in range(epoch):
                # training
                start_time = time.time()
                md.train()
                if i == 0:
                    training_time.append(time.time()-start_time)
                else:
                    training_time.append(time.time()-start_time+training_time[-1])
                print("-- Finish epoch {0:d}".format(i+1))
                print '==>>> epoch: {}, training time: {:.6f}'.format(i+1, training_time[-1])
                # training error
                print '==>>> verify training error'
                for batch_idx, (x, target) in enumerate(train_loader):
                    x_pred = md.forward(Variable(x))
                    x_pred = np.argmax(x_pred.data.numpy(), 1)
                    accu = accuracy_score(target.numpy(), x_pred)
                    if batch_idx == len(test_loader)-1:
                        print '==>>> epoch: {}, training error: {:.6f}'.format(i+1, 1-accu)
                        training_errors.append(1-accu)
                # testing error
                print '==>>> verify testing error'
                for batch_idx, (x, target) in enumerate(test_loader):
                    x_pred = md.forward(Variable(x))
                    x_pred = np.argmax(x_pred.data.numpy(), 1)
                    accu = accuracy_score(target.numpy(), x_pred)
                    if batch_idx == len(test_loader)-1:
                        print '==>>> epoch: {}, testing error: {:.6f}'.format(i+1, 1-accu)
                        testing_errors.append(1-accu)

                plt.title(type(model()).__name__)
                plt.xlabel("epochs")
                plt.ylabel("error")
                plt.plot(range(i+1), training_errors, "bo", range(i+1), training_errors, "b--", label="training_error")
                plt.plot(range(i+1), testing_errors, "ro", range(i+1), testing_errors, "r--", label="testing_error")
                plt.legend(loc='upper right')
                plt.xticks(range(i+1))
                plt.savefig("{0}_error_ep-{1:d}.png".format(type(model()).__name__, i+1))
                plt.clf()

                plt.title(type(model()).__name__)
                plt.xlabel("epochs")
                plt.ylabel("seconds")
                plt.plot(range(i+1), training_time, "bo", range(i+1), training_time, "b--", label="training_time")
                plt.legend(loc='upper left')
                plt.xticks(range(i+1))
                plt.savefig("{0}_speed_ep-{1:d}.png".format(type(model()).__name__, i))
                plt.clf()
            print("Done!")

if __name__ == "__main__":
    main()
