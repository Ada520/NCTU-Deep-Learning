from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

def to_var(x, volatile=False):
    return Variable(x, volatile=volatile)

def idx2onehot(idx, n):
    onehot = torch.zeros(idx.size(0), n).cuda()
    onehot.scatter_(1, idx.data, 1)
    onehot = to_var(onehot)
    return onehot

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
#lr=1e-3
#noise size  = 20
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.en_conv0 = nn.Conv2d(11, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.en_conv1 = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.fc1 = nn.Linear(784, 400, bias=True)
        self.fc21 = nn.Linear(400, 20, bias=True)
        self.fc22 = nn.Linear(400, 20, bias=True)
        self.fc3 = nn.Linear(30, 392, bias=True)
        
        self.de_conv0 = nn.Conv2d(2, 11, kernel_size=3, stride=1, padding=1, bias=False)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.de_conv1 = nn.Conv2d(11, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.de_conv2 = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        out = self.relu(self.en_conv0(x))
        out = self.relu(self.en_conv1(out))
        out = out.view(out.size(0), -1)
        h1 = self.relu(self.fc1(out))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar).cuda()
            eps = Variable(torch.randn(std.size())).cuda()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        h3.data.resize_(h3.size(0), 2, 14, 14)
        h3 = self.up(self.relu(self.de_conv0(h3)))
        h3 = self.relu(self.de_conv1(h3))
        return self.sigmoid(self.de_conv2(h3))

    def forward(self, x, c = None):
        one_hot = idx2onehot(c, n=10)
        for i in range(28):
            for j in range(28):
                tmp = torch.cat((x[:, :, i, j], one_hot), dim=-1)
                if i==0 and j==0:
                    e_input=tmp
                else:
                    e_input = torch.cat((e_input, tmp), dim=-1)
        e_input.data.resize_(e_input.size(0), 11, 28, 28)
        mu, logvar = self.encode(e_input)
        z = self.reparameterize(mu, logvar)
        z = torch.cat((z, one_hot), dim=-1)
        return self.decode(z), mu, logvar


model = VAE()
if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x.view(-1, 784), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = to_var(data).cuda()
        target = to_var(target).cuda()
        target = target.view(-1, 1)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data,target)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    return train_loss / len(train_loader.dataset)


def test(epoch):
    model.eval()
    test_loss = 0
    for i, (data, target) in enumerate(test_loader):

        data = to_var(data).cuda()
        target = to_var(target).cuda()
        target = target.view(-1, 1)

        recon_batch, mu, logvar = model(data,target)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                  recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(),
                     'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

loss_list= []
for epoch in range(1, args.epochs + 1):
    loss_list.append(train(epoch))
    test(epoch)
    c = to_var(torch.arange(0,10).long().view(-1,1)).cuda()
    c = idx2onehot(c, n=10)
    tmp = c
    for i in range(1,10):
        tmp = torch.cat((tmp, c), dim=-1)
    c_s = torch.transpose(tmp, 0, 1)
    sample = to_var(torch.randn(20, 10, 1)).cuda()
    tmp = sample
    for i in range(1,10):
        tmp = torch.cat((tmp, sample), dim=-1)
    s_s = torch.transpose(tmp, 1, 2)
    s_s.data.resize_(20, 100)
    s_s = torch.transpose(s_s, 0, 1)
    sample = Variable(torch.randn(100, 20))
    #noise size = 20, condition code size = 10
    sample = torch.cat((s_s, c_s), dim=-1)
    sample = model.decode(sample).cpu()
    save_image(sample.data.view(100, 1, 28, 28),
'results/sample_' + str(epoch) + '.png',nrow=10)
    torch.save(model.state_dict(), 'results/cvae_%d.pth' % (epoch))

plt.title('training loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(np.arange(1,len(loss_list)+1),loss_list , color = 'blue')
savefilename = 'cvae_training_loss.jpg'
plt.savefig(savefilename)
plt.show()
plt.close()
