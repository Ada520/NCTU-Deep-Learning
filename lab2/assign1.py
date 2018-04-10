import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
import PIL
from utils import *

reg_noise_std = 1./30.
sigma = 25
sigma_ = sigma/255.
num_steps = 2401
PLOT = True
dtype = torch.cuda.FloatTensor

class deep_image_prior(nn.Module):
    def __init__(self):
        super(deep_image_prior, self).__init__()
        self.conv3_8 = nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv8_8 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.conv8_16 = nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.conv16_16 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv16_32 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv32_32 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv32_64 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv64_64 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv64_128 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv128_128 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv128_64 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.conv64_32 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.conv32_16 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.conv16_8 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.conv8_3 = nn.Conv2d(8, 3, 3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.MaxPool2d = nn.MaxPool2d(2, 2)

        self.bn2d_8 = nn.BatchNorm2d(8)
        self.bn2d_16 = nn.BatchNorm2d(16)
        self.bn2d_32 = nn.BatchNorm2d(32)
        self.bn2d_64 = nn.BatchNorm2d(64)
        self.bn2d_128 = nn.BatchNorm2d(128)
        self.bn2d_3 = nn.BatchNorm2d(3)
        self.skip64 = nn.Conv2d(64, 4, 1, stride=1, padding=0)
        self.skip128 = nn.Conv2d(128, 4, 1, stride=1, padding=0)

        self.bn2d_68 = nn.BatchNorm2d(68)
        self.conv68_64 = nn.Conv2d(68, 64, 3, stride=1, padding=1)
        self.bn2d_132 = nn.BatchNorm2d(132)
        self.conv132_128 = nn.Conv2d(132, 128, 3, stride=1, padding=1)
        self.up_bilinear = nn.Upsample(scale_factor=2, mode='bilinear')

        
        
        
    def forward(self, noise):
        down_1 = self.conv3_8(noise)
        down_1 = self.MaxPool2d(down_1)
        down_1 = self.bn2d_8(down_1)
        down_1 = F.leaky_relu(down_1)
        down_1 = self.conv8_8(down_1)
        down_1 = self.bn2d_8(down_1)
        down_1 = F.leaky_relu(down_1)
        
        down_2 = self.conv8_16(down_1)
        down_2 = self.MaxPool2d(down_2)
        down_2 = self.bn2d_16(down_2)
        down_2 = F.leaky_relu(down_2)
        down_2 = self.conv16_16(down_2)
        down_2 = self.bn2d_16(down_2)
        down_2 = F.leaky_relu(down_2)

        down_3 = self.conv16_32(down_2)
        down_3 = self.MaxPool2d(down_3)
        down_3 = self.bn2d_32(down_3)
        down_3 = F.leaky_relu(down_3)
        down_3 = self.conv32_32(down_3)
        down_3 = self.bn2d_32(down_3)
        down_3 = F.leaky_relu(down_3)


        down_4 = self.conv32_64(down_3)
        down_4 = self.MaxPool2d(down_4)
        down_4 = self.bn2d_64(down_4)
        down_4 = F.leaky_relu(down_4)
        down_4 = self.conv64_64(down_4)
        down_4 = self.bn2d_64(down_4)
        down_4 = F.leaky_relu(down_4)
        skip_4 = self.skip64(down_4)

        down_5 = self.conv64_128(down_4)
        down_5 = self.MaxPool2d(down_5)
        down_5 = self.bn2d_128(down_5)
        down_5 = F.leaky_relu(down_5)
        down_5 = self.conv128_128(down_5)
        down_5 = self.bn2d_128(down_5)
        down_5 = F.leaky_relu(down_5)
        skip_5 = self.skip128(down_5)

        up_5 = torch.cat([down_5, skip_5], 1)
        up_5 = self.bn2d_132(up_5)
        up_5 = self.conv132_128(up_5)
        up_5 = self.bn2d_128(up_5)
        up_5 = F.leaky_relu(up_5)
        up_5 = self.conv128_64(up_5)
        up_5 = self.bn2d_64(up_5)
        up_5 = F.leaky_relu(up_5)
        up_5 = self.up_bilinear(up_5)

        up_4 = torch.cat([up_5, skip_4], 1)
        up_4 = self.bn2d_68(up_4)
        up_4 = self.conv68_64(up_4)
        up_4 = self.bn2d_64(up_4)
        up_4 = F.leaky_relu(up_4)
        up_4 = self.conv64_32(up_4)
        up_4 = self.bn2d_32(up_4)
        up_4 = F.leaky_relu(up_4)
        up_4 = self.up_bilinear(up_4)

        up_3 = self.bn2d_32(up_4)
        up_3 = self.conv32_32(up_3)
        up_3 = self.bn2d_32(up_3)
        up_3 = F.leaky_relu(up_3)
        up_3 = self.conv32_16(up_3)
        up_3 = self.bn2d_16(up_3)
        up_3 = F.leaky_relu(up_3)
        up_3 = self.up_bilinear(up_3)

        up_2 = self.bn2d_16(up_3)
        up_2 = self.conv16_16(up_2)
        up_2 = self.bn2d_16(up_2)
        up_2 = F.leaky_relu(up_2)
        up_2 = self.conv16_8(up_2)
        up_2 = self.bn2d_8(up_2)
        up_2 = F.leaky_relu(up_2)
        up_2 = self.up_bilinear(up_2)

        up_1 = self.bn2d_8(up_2)
        up_1 = self.conv8_8(up_1)
        up_1 = self.bn2d_8(up_1)
        up_1 = F.leaky_relu(up_1)
        up_1 = self.conv8_3(up_1)
        up_1 = self.bn2d_3(up_1)
        up_1 = F.leaky_relu(up_1)
        up_1 = self.up_bilinear(up_1)

        out = self.conv3_3(up_1)
        out = self.bn2d_3(out)
        out = F.sigmoid(out)

        return out

imgs_path = 'images/cat.jpg'

imgs = crop_image(get_image(imgs_path, -1)[0], d=32)
imgs_np = pil_to_np(imgs)

imgs_var = np_to_var(imgs_np).type(dtype)

imgs_noisy_pil, imgs_noisy_np = get_noisy_image(imgs_np, sigma_)
imgs_noisy_var = np_to_var(imgs_noisy_np).type(dtype)

imgs_np_var = np_to_var(imgs_np).type(dtype)
shuffle = shuffle_tensor(imgs_np_var.data)

noise_img = get_noise(3, 'noise', (imgs.size[1], imgs.size[0]), noise_type='u', var = 1).type(dtype).detach()

net_input = get_noise(3, 'noise', (imgs.size[1], imgs.size[0]), noise_type='u').type(dtype).detach()
net_input_saved = net_input.data.clone()
noise = net_input.data.clone()

shuffle_var = Variable(shuffle)
shuffle_var = var_to_np(shuffle_var)
shuffle_var = np_to_var(shuffle_var).type(dtype)

tensor_to_jpg(imgs_np_var.data, "output/assign1/GT_1.jpg")
tensor_to_jpg(imgs_noisy_var.data, "output/assign1/GT_2.jpg")
tensor_to_jpg(shuffle_var.data, "output/assign1/GT_3.jpg")
tensor_to_jpg(net_input.data, "output/assign1/GT_4.jpg")

if __name__=='__main__':
    
    print('normal image:')
    net1 = deep_image_prior()
    optimizer1 = torch.optim.Adam(net1.parameters(), lr=1e-2)
    criterion1 = nn.MSELoss().type(dtype)
    
    net1.cuda()
    loss1 =[]
    for step in range(num_steps):
        if reg_noise_std > 0:
            net_input.data = net_input_saved + (noise.normal_() * reg_noise_std)

        output = net1(net_input)
        optimizer1.zero_grad()
        loss = criterion1(output, imgs_var)
        loss1.append(loss.data[0])
        loss.backward()
        optimizer1.step()

        if PLOT and step % 100 == 0:
            print ('Iteration %04d    Loss %.4f' % (step,loss.data[0]))
            out_np = var_to_np(output)
            savefilename = 'output/assign1/1_output' + str(step) + ".jpg"
            plot_image_grid([np.clip(out_np, 0, 1)],fig_name=savefilename)

##############################################################################
    print('image + noise:')
    net2 = deep_image_prior()
    optimizer2 = torch.optim.Adam(net2.parameters(), lr=1e-2)
    criterion2 = nn.MSELoss().type(dtype)
    
    net2.cuda()
    loss2 =[]
    for step in range(num_steps):
        if reg_noise_std > 0:
            net_input.data = net_input_saved + (noise.normal_() * reg_noise_std)

        output = net2(net_input)
        optimizer2.zero_grad()
        loss = criterion2(output, imgs_noisy_var)
        loss2.append(loss.data[0])
        loss.backward()
        optimizer2.step()

        if PLOT and step % 100 == 0:
            print ('Iteration %04d    Loss %.4f' % (step,loss.data[0]))
            out_np = var_to_np(output)
            savefilename = 'output/assign1/2_output' + str(step) + ".jpg"
            plot_image_grid([np.clip(out_np, 0, 1)],fig_name=savefilename)

##############################################################################
    print('shuffled image:')
    net3 = deep_image_prior()
    optimizer3 = torch.optim.Adam(net3.parameters(), lr=1e-2)
    criterion3 = nn.MSELoss().type(dtype)
    
    net3.cuda()
    loss3 =[]
    for step in range(num_steps):
        if reg_noise_std > 0:
            net_input.data = net_input_saved + (noise.normal_() * reg_noise_std)

        output = net3(net_input)
        optimizer3.zero_grad()
        loss = criterion3(output, shuffle_var)
        loss3.append(loss.data[0])
        loss.backward()
        optimizer3.step()

        if PLOT and step % 100 == 0:
            print ('Iteration %04d    Loss %.4f' % (step,loss.data[0]))
            out_np = var_to_np(output)
            savefilename = 'output/assign1/3_output' + str(step) + ".jpg"
            plot_image_grid([np.clip(out_np, 0, 1)],fig_name=savefilename)
    
##############################################################################
    print('random noise:')
    net4 = deep_image_prior()
    optimizer4 = torch.optim.Adam(net4.parameters(), lr=1e-2)
    criterion4 = nn.MSELoss().type(dtype)
    
    net4.cuda()
    loss4 =[]
    for step in range(num_steps):
        if reg_noise_std > 0:
            net_input.data = net_input_saved + (noise.normal_() * reg_noise_std)

        output = net4(net_input)
        optimizer4.zero_grad()
        loss = criterion4(output, noise_img)
        loss4.append(loss.data[0])
        loss.backward()
        optimizer4.step()

        if PLOT and step % 100 == 0:
            print ('Iteration %04d    Loss %.4f' % (step,loss.data[0]))
            out_np = var_to_np(output)
            savefilename = 'output/assign1/4_output' + str(step) + ".jpg"
            plot_image_grid([np.clip(out_np, 0, 1)],fig_name=savefilename)

plt.title('image loss')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.ylim(0, 0.1)
plt.xlim(1, 2500)
plt.plot(np.arange(1,len(loss1)+1),loss1 , color = 'blue',label='normal img')
plt.plot(np.arange(1,len(loss2)+1),loss2 , color = 'green',label='noisy img')
plt.plot(np.arange(1,len(loss3)+1),loss3, color = 'red',label='shuffled img')
plt.plot(np.arange(1,len(loss4)+1),loss4 , color = 'yellow',label='noise')
plt.legend(loc='upper right')
savefilename = 'assign1.jpg'
plt.savefig(savefilename)
plt.show()
plt.close()
