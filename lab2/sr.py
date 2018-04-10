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
num_steps = 2001
PLOT = True
dtype = torch.cuda.FloatTensor


class deep_image_prior(nn.Module):
    def __init__(self):
        super(deep_image_prior, self).__init__()
        self.conv32_128 = nn.Conv2d(32, 128, 3, stride=1, padding=1)
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
        down_1 = self.conv32_128(noise)
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

ground_truth_path = 'images/SR_GT.png'
low_r_path = 'images/LowResolution.png'
imgs_L, imgs_L_np = get_image(low_r_path)
imgs_H, imgs_H_np = get_image(ground_truth_path)

net_input = get_noise(32, 'noise', (imgs_H.size[1], imgs_H.size[0])).type(dtype).detach()
net_input_saved = net_input.data.clone()
noise = net_input.data.clone()

img_LR_var = np_to_var(imgs_L_np).type(dtype)
'''
imgs = load_LR_HR_imgs_sr(ground_truth_path , -1, 4, 'CROP')

net_input = get_noise(32, 'noise', (imgs['HR_pil'].size[1], imgs['HR_pil'].size[0])).type(dtype).detach()
img_LR_var = np_to_var(imgs['LR_np']).type(dtype)
'''
downsampler = Downsampler(n_planes=3, factor=4, kernel_type='lanczos2', phase=0.5, preserve_size=True).type(dtype)

if __name__=='__main__':
####high resolution#####
    net = deep_image_prior()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    criterion = nn.MSELoss().type(dtype)
    
    net.cuda()

    for step in range(num_steps):
        if reg_noise_std > 0:
            net_input.data = net_input_saved + (noise.normal_() * reg_noise_std)

        out_hr = net(net_input)
        out_lr = downsampler(out_hr)
        optimizer.zero_grad()
        loss = criterion(out_lr, img_LR_var)
        loss.backward()
        optimizer.step()

        if PLOT and step % 100 == 0:
            out_hr_np = var_to_np(out_hr)
            print("Iteration: %04d    Loss: %.4f    PSNR: %.4f "% (step,loss.data[0],compare_psnr(imgs_H_np, out_hr_np)) )
            savefilename = 'output/sr/high_resolution_output' + str(step) + ".jpg"
            plot_image_grid([np.clip(out_hr_np, 0, 1)],fig_name=savefilename)

