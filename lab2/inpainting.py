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
num_steps = 5001
PLOT = True
dtype = torch.cuda.FloatTensor


class deep_image_prior(nn.Module):
    def __init__(self):
        super(deep_image_prior, self).__init__()
        self.conv32_128 = nn.Conv2d(2, 128, 3, stride=1, padding=1)
        self.MaxPool2d = nn.MaxPool2d(2, 2)
        self.bn2d_128 = nn.BatchNorm2d(128)
        self.conv128_128 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.up_nearest = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv128_3 = nn.Conv2d(128, 3, 3, stride=1, padding=1)
        self.bn2d_3 = nn.BatchNorm2d(3)
        
    def forward(self, noise):
        down_1 = self.conv32_128(noise)
        down_1 = self.MaxPool2d(down_1)
        down_1 = self.bn2d_128(down_1)
        down_1 = F.leaky_relu(down_1)
        down_1 = self.conv128_128(down_1)
        down_1 = self.bn2d_128(down_1)
        down_1 = F.leaky_relu(down_1)
        
        down_2 = self.conv128_128(down_1)
        down_2 = self.MaxPool2d(down_2)
        down_2 = self.bn2d_128(down_2)
        down_2 = F.leaky_relu(down_2)
        down_2 = self.conv128_128(down_2)
        down_2 = self.bn2d_128(down_2)
        down_2 = F.leaky_relu(down_2)

        down_3 = self.conv128_128(down_2)
        down_3 = self.MaxPool2d(down_3)
        down_3 = self.bn2d_128(down_3)
        down_3 = F.leaky_relu(down_3)
        down_3 = self.conv128_128(down_3)
        down_3 = self.bn2d_128(down_3)
        down_3 = F.leaky_relu(down_3)

        down_4 = self.conv128_128(down_3)
        down_4 = self.MaxPool2d(down_4)
        down_4 = self.bn2d_128(down_4)
        down_4 = F.leaky_relu(down_4)
        down_4 = self.conv128_128(down_4)
        down_4 = self.bn2d_128(down_4)
        down_4 = F.leaky_relu(down_4)

        down_5 = self.conv128_128(down_4)
        down_5 = self.MaxPool2d(down_5)
        down_5 = self.bn2d_128(down_5)
        down_5 = F.leaky_relu(down_5)
        down_5 = self.conv128_128(down_5)
        down_5 = self.bn2d_128(down_5)
        down_5 = F.leaky_relu(down_5)

        up_5 = self.conv128_128(down_5)
        up_5 = self.bn2d_128(up_5)
        up_5 = F.leaky_relu(up_5)
        up_5 = self.conv128_128(up_5)
        up_5 = self.bn2d_128(up_5)
        up_5 = F.leaky_relu(up_5)
        up_5 = self.up_nearest(up_5)


        up_4 = self.conv128_128(up_5)
        up_4 = self.bn2d_128(up_4)
        up_4 = F.leaky_relu(up_4)
        up_4 = self.conv128_128(up_4)
        up_4 = self.bn2d_128(up_4)
        up_4 = F.leaky_relu(up_4)
        up_4 = self.up_nearest(up_4)



        up_3 = self.conv128_128(up_4)
        up_3 = self.bn2d_128(up_3)
        up_3 = F.leaky_relu(up_3)
        up_3 = self.conv128_128(up_3)
        up_3 = self.bn2d_128(up_3)
        up_3 = F.leaky_relu(up_3)
        up_3 = self.up_nearest(up_3)

        up_2 = self.conv128_128(up_3)
        up_2 = self.bn2d_128(up_2)
        up_2 = F.leaky_relu(up_2)
        up_2 = self.conv128_128(up_2)
        up_2 = self.bn2d_128(up_2)
        up_2 = F.leaky_relu(up_2)
        up_2 = self.up_nearest(up_2)

        up_1 = self.conv128_128(up_2)
        up_1 = self.bn2d_128(up_1)
        up_1 = F.leaky_relu(up_1)
        up_1 = self.conv128_128(up_1)
        up_1 = self.bn2d_128(up_1)
        up_1 = F.leaky_relu(up_1)
        up_1 = self.up_nearest(up_1)

        out = self.conv128_3(up_1)
        out = self.bn2d_3(out)
        out = F.sigmoid(out)

        return out

ground_truth_path = 'images/bonus/2.png'
mask_path = 'images/bonus/2_mask.png'
imgs, imgs_np = get_image(ground_truth_path)
mask, mask_np = get_image(mask_path)

mask = crop_image(mask,64)
imgs = crop_image(imgs,64)

imgs_np = pil_to_np(imgs)
mask_np = pil_to_np(mask)

net_input = get_noise(2, 'meshgrid', imgs_np.shape[1:]).type(dtype).detach()
net_input_saved = net_input.data.clone()
noise = net_input.data.clone()

imgs_var = np_to_var(imgs_np).type(dtype)
mask_var = np_to_var(mask_np).type(dtype)

if __name__=='__main__':
    net = deep_image_prior()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    criterion = nn.MSELoss().type(dtype)
    
    net.cuda()

    for step in range(num_steps):
        if reg_noise_std > 0:
            net_input.data = net_input_saved + (noise.normal_() * reg_noise_std)

        out = net(net_input)

        optimizer.zero_grad()
        loss = criterion(out * mask_var, imgs_var * mask_var)
        loss.backward()
        optimizer.step()

        if PLOT and step % 100 == 0:
            out_np = var_to_np(out)
            print("Iteration: %04d    Loss: %.4f "% (step,loss.data[0]) )
            savefilename = 'output/inpainting/ip_output' + str(step) + ".jpg"
            plot_image_grid([np.clip(out_np, 0, 1)],fig_name=savefilename)

