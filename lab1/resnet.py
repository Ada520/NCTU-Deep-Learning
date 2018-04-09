from __future__ import print_function
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='PyTorch CNN')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=164, 
                    help='number of epochs to train (default: 164)')
parser.add_argument('--lr', type=float, default=0.1, 
                    help='learning rate (default: 0.1)')
#Initial learning rate: 0.1, divide by 10 at 81, 122 epoch
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=0.0001, 
                    help='momentum Weight_decay (default: 0.0001)')
parser.add_argument('--block_types', type=int, default=1, 
                    help='choose network type ,1: ResNet, 2: Pre-act ResNet, 3: Vanilla CNN (default:1)')
parser.add_argument('--blocks', type=int, default=3, 
                    help='blocks (default: 3 (20 layers) )')

args = parser.parse_args()

#torch.cuda.manual_seed(1)

#Weight initialization: torch.nn.init.kaiming_normal
#Loss function: cross-entropy

# Dataloader
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
                       transforms.RandomHorizontalFlip(),
	                   transforms.Pad(4, fill=0),
	                   transforms.RandomCrop(32),
                       transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4824, 0.4467), (0.2471, 0.2435, 0.2616)),
                   ])),
    batch_size=args.batch_size, shuffle=True,num_workers = 2)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4824, 0.4467), (0.2471, 0.2435, 0.2616))
                   ])),
    batch_size=args.batch_size, shuffle=True,num_workers = 2)

# 3x3 Convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class PreActBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(PreActBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)	# for bonus, pre-act ResNet 
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        return out

class VanillaBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(VanillaBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out

# ResNet Module
class Net(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(Net, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0],2)
        self.layer3 = self.make_layer(block, 64, layers[0],2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')


    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
            	nn.MaxPool2d(2,2),
                conv3x3(self.in_channels, out_channels),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


if args.block_types == 1:
    Net = Net(ResidualBlock, [args.blocks])	#ResNet
    filename = 'ResNet ' + str(args.blocks*6+2)
elif args.block_types == 2:
	Net = Net(PreActBlock, [args.blocks])	#for bonus, Pre-act ResNet
	filename = 'Pre-act ResNet ' + str(args.blocks*6+2)
else:
	Net = Net(VanillaBlock, [args.blocks])	#vanilla cnn
	filename = 'Vanilla CNN ' + str(args.blocks*6+2)

print (filename)
Net.cuda()
Loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(Net.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.weight_decay)


def adjust_learning_rate(optimizer, epoch):

    if epoch < 81:
       lr = 0.1
    elif epoch < 122:
       lr = 0.01
    else: 
       lr = 0.001

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

trainl = []
def train(epoch):
    Net.train()
    adjust_learning_rate(optimizer, epoch)
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = Net(data)
        loss = Loss(output, target)
        train_loss += Loss(output, target).data[0]
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    train_loss = train_loss
    train_loss /= len(train_loader)
    trainl.append(train_loss)

tl = []
te = []
def test(epoch):
    Net.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = Net(data)
        test_loss += Loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    tl.append(test_loss)
    test_error = (10000.0 - correct) / len(test_loader.dataset)
    te.append(test_error)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(args.epochs)
    '''
    savefilename = 'CNN_'+str(epoch)+'.tar'
    torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
        }, savefilename)
	'''
fig = plt.figure()
trainplt = fig.add_subplot(2,1,1)
trainplt.set_ylabel('loss')
trainplt.set_ylim((0,2.5))
trainplt.set_xlabel('epochs')
trainplt.set_title('training loss')
trainplt.plot(np.arange(1,len(tl)+1),trainl)

testplt = fig.add_subplot(2,1,2)
testplt.set_ylabel('error')
testplt.set_ylim((0,1))
testplt.set_xlabel('epochs')
testplt.set_title('test error')
testplt.plot(np.arange(1,len(te)+1),te , color = 'red')

plt.tight_layout()
plt.savefig(filename+'.png')
plt.show()

'''
plt.ylabel('loss')
plt.xlabel('epochs')
plt.title('model loss')
trainplt, = plt.plot(np.arange(1,len(tl)+1),trainl)
#testplt, = plt.plot(np.arange(1,len(tl)+1),tl,color='red')
plt.legend((trainplt), ('training loss'))
plt.savefig(filename+'.png')
plt.show()
'''