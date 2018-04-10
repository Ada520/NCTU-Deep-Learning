import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.measure import compare_psnr
from PIL import Image
import PIL
from utils import *

reg_noise_std = 1./30.
sigma = 25
sigma_ = sigma/255.
num_steps = 1801
PLOT = True
dtype = torch.cuda.FloatTensor


class deep_image_prior(nn.Module):
    def __init__(self):
        super(deep_image_prior, self).__init__()
        self.conv3_128 = nn.Conv2d(3, 128, 3, stride=1, padding=1)
        self.MaxPool2d = nn.MaxPool2d(2, 2)
        self.bn2d_128 = nn.BatchNorm2d(128)
        self.conv128_128 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.skip = nn.Conv2d(128, 4, 1, stride=1, padding=0)

        self.bn2d_132 = nn.BatchNorm2d(132)
        self.conv132_128 = nn.Conv2d(132, 128, 3, stride=1, padding=1)
        self.up_bilinear = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv128_3 = nn.Conv2d(128, 3, 3, stride=1, padding=1)
        self.bn2d_3 = nn.BatchNorm2d(3)
        
    def forward(self, noise):
        down_1 = self.conv3_128(noise)
        down_1 = self.MaxPool2d(down_1)
        down_1 = self.bn2d_128(down_1)
        down_1 = F.leaky_relu(down_1)
        down_1 = self.conv128_128(down_1)
        down_1 = self.bn2d_128(down_1)
        down_1 = F.leaky_relu(down_1)
        skip_1 = self.skip(down_1)
        
        down_2 = self.conv128_128(down_1)
        down_2 = self.MaxPool2d(down_2)
        down_2 = self.bn2d_128(down_2)
        down_2 = F.leaky_relu(down_2)
        down_2 = self.conv128_128(down_2)
        down_2 = self.bn2d_128(down_2)
        down_2 = F.leaky_relu(down_2)
        skip_2 = self.skip(down_2)

        down_3 = self.conv128_128(down_2)
        down_3 = self.MaxPool2d(down_3)
        down_3 = self.bn2d_128(down_3)
        down_3 = F.leaky_relu(down_3)
        down_3 = self.conv128_128(down_3)
        down_3 = self.bn2d_128(down_3)
        down_3 = F.leaky_relu(down_3)
        skip_3 = self.skip(down_3)

        down_4 = self.conv128_128(down_3)
        down_4 = self.MaxPool2d(down_4)
        down_4 = self.bn2d_128(down_4)
        down_4 = F.leaky_relu(down_4)
        down_4 = self.conv128_128(down_4)
        down_4 = self.bn2d_128(down_4)
        down_4 = F.leaky_relu(down_4)
        skip_4 = self.skip(down_4)

        down_5 = self.conv128_128(down_4)
        down_5 = self.MaxPool2d(down_5)
        down_5 = self.bn2d_128(down_5)
        down_5 = F.leaky_relu(down_5)
        down_5 = self.conv128_128(down_5)
        down_5 = self.bn2d_128(down_5)
        down_5 = F.leaky_relu(down_5)
        skip_5 = self.skip(down_5)

        up_5 = torch.cat([down_5, skip_5], 1)
        up_5 = self.bn2d_132(up_5)
        up_5 = self.conv132_128(up_5)
        up_5 = self.bn2d_128(up_5)
        up_5 = F.leaky_relu(up_5)
        up_5 = self.conv128_128(up_5)
        up_5 = self.bn2d_128(up_5)
        up_5 = F.leaky_relu(up_5)
        up_5 = self.up_bilinear(up_5)

        up_4 = torch.cat([up_5, skip_4], 1)
        up_4 = self.bn2d_132(up_4)
        up_4 = self.conv132_128(up_4)
        up_4 = self.bn2d_128(up_4)
        up_4 = F.leaky_relu(up_4)
        up_4 = self.conv128_128(up_4)
        up_4 = self.bn2d_128(up_4)
        up_4 = F.leaky_relu(up_4)
        up_4 = self.up_bilinear(up_4)

        up_3 = torch.cat([up_4, skip_3], 1)
        up_3 = self.bn2d_132(up_3)
        up_3 = self.conv132_128(up_3)
        up_3 = self.bn2d_128(up_3)
        up_3 = F.leaky_relu(up_3)
        up_3 = self.conv128_128(up_3)
        up_3 = self.bn2d_128(up_3)
        up_3 = F.leaky_relu(up_3)
        up_3 = self.up_bilinear(up_3)

        up_2 = torch.cat([up_3, skip_2], 1)
        up_2 = self.bn2d_132(up_2)
        up_2 = self.conv132_128(up_2)
        up_2 = self.bn2d_128(up_2)
        up_2 = F.leaky_relu(up_2)
        up_2 = self.conv128_128(up_2)
        up_2 = self.bn2d_128(up_2)
        up_2 = F.leaky_relu(up_2)
        up_2 = self.up_bilinear(up_2)

        up_1 = torch.cat([up_2, skip_1], 1)
        up_1 = self.bn2d_132(up_1)
        up_1 = self.conv132_128(up_1)
        up_1 = self.bn2d_128(up_1)
        up_1 = F.leaky_relu(up_1)
        up_1 = self.conv128_128(up_1)
        up_1 = self.bn2d_128(up_1)
        up_1 = F.leaky_relu(up_1)
        up_1 = self.up_bilinear(up_1)

        out = self.conv128_3(up_1)
        out = self.bn2d_3(out)
        out = F.sigmoid(out)

        return out

ground_truth_path = 'images/noise_GT.png'
imgs_path = 'images/noise_image.png'
'''
imgs_GT = crop_image(get_image(ground_truth_path, -1)[0], d=32)
imgs_GT_np = pil_to_np(imgs_GT)
'''
imgs_GT, imgs_GT_np = get_image(ground_truth_path)
'''
imgs = crop_image(get_image(imgs_path, -1)[0], d=32)
imgs_np = pil_to_np(imgs)
'''
imgs, imgs_np = get_image(imgs_path)


net_input = get_noise(3, 'noise', (imgs.size[1], imgs.size[0]), noise_type='u',var = 1.).type(dtype).detach()
net_input_saved = net_input.data.clone()
noise = net_input.data.clone()

img_var = np_to_var(imgs_np).type(dtype)

if __name__=='__main__':
    net = deep_image_prior()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    criterion = nn.MSELoss().type(dtype)
    
    net.cuda()

    for step in range(num_steps):
        
        if reg_noise_std > 0:
            net_input.data = net_input_saved + (noise.normal_() * reg_noise_std)
        
        output = net(net_input)
        optimizer.zero_grad()
        loss = criterion(output, img_var)
        loss.backward()
        optimizer.step()

        if PLOT and step % 100 == 0:
            out_np = var_to_np(output)
            print("Iteration: %04d    Loss: %.4f    PSNR: %.4f "% (step,loss.data[0],compare_psnr(imgs_GT_np, out_np)) )
            savefilename = 'output/denoising/denoising_output' + str(step) + ".jpg"
            plot_image_grid([np.clip(out_np, 0, 1)],fig_name=savefilename)