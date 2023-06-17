import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

# Use torchvision to load data
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())
# length of train and test data
print("the length of train dataset is:{}".format(len(train_data)))  # 50000
print("the length of test dataset is:{}".format(len(test_data)))  # 10000
# train_data[0] is a tuple of image and label
print("the type of train_data[0] is:{}".format(type(train_data[0])))  # tuple
print("the length of train_data[0] is:{}".format(len(train_data[0])))  # 2
# train_data[0][0] is a tensor of image
# train_data[0][1] is a int of label

# Use dataloader to load data
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Bulid a model
#no
