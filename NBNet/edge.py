import numpy as np
import torch
import torch.nn as nn
import math
from scipy.signal.windows import gaussian


def get_filter(filter_size=5, std=1.0, map_func=lambda x:x):
    generated_filters = gaussian(filter_size, std=std).reshape([1, filter_size]).astype(np.float32)

    gaussian_filter_horizontal = generated_filters[None, None, ...]

    gaussian_filter_vertical = generated_filters.T[None, None, ...]

    sobel_filter_horizontal = np.array([[[
        [1., 0., -1.],
        [2., 0., -2.],
        [1., 0., -1.]]]],
        dtype='float32'
    )

    sobel_filter_vertical = np.array([[[
        [1., 2., 1.],
        [0., 0., 0.],
        [-1., -2., -1.]]]],
        dtype='float32'
    )

    directional_filter = np.array(
        [[[[ 0.,  0.,  0.],
          [ 0.,  1., -1.],
          [ 0.,  0.,  0.]]],


        [[[ 0.,  0.,  0.],
          [ 0.,  1.,  0.],
          [ 0.,  0., -1.]]],


        [[[ 0.,  0.,  0.],
          [ 0.,  1.,  0.],
          [ 0., -1.,  0.]]],


        [[[ 0.,  0.,  0.],
          [ 0.,  1.,  0.],
          [-1.,  0.,  0.]]],


        [[[ 0.,  0.,  0.],
          [-1.,  1.,  0.],
          [ 0.,  0.,  0.]]],


        [[[-1.,  0.,  0.],
          [ 0.,  1.,  0.],
          [ 0.,  0.,  0.]]],


        [[[ 0., -1.,  0.],
          [ 0.,  1.,  0.],
          [ 0.,  0.,  0.]]],


        [[[ 0.,  0., -1.],
          [ 0.,  1.,  0.],
          [ 0.,  0.,  0.]]]],
        dtype=np.float32
    )

    connect_filter = np.array([[[
        [1., 1., 1.],
        [1., 0., 1.],
        [1., 1., 1.]]]],
        dtype=np.float32
    )

    return {
        'gaussian_filter_horizontal.weight': map_func(gaussian_filter_horizontal),
        'gaussian_filter_vertical.weight': map_func(gaussian_filter_vertical),
        'sobel_filter_horizontal.weight': map_func(sobel_filter_horizontal),
        'sobel_filter_vertical.weight': map_func(sobel_filter_vertical),
        'directional_filter.weight': map_func(directional_filter),
        'connect_filter.weight': map_func(connect_filter)
    }


class CannyDetector(nn.Module):
    def __init__(self, filter_size=5, std=1.0, device='cpu', detach=False):
        super(CannyDetector, self).__init__()

        # 配置运行设备
        self.device = device
        self.detach = detach
        # 高斯滤波器
        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2), bias=False)
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0), bias=False)

        # Sobel 滤波器
        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)

        # 定向滤波器
        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, bias=False)

        # 连通滤波器
        self.connect_filter = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)

        # 初始化参数
        params = get_filter(filter_size=filter_size, std=std, map_func=lambda x:torch.from_numpy(x).to(self.device))
        self.load_state_dict(params)

        # parameters
        self.w = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.b = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w.data.fill_(1)
        self.b.data.fill_(0)

        self.idx = 0

    def forward(self, img, threshold1=10.0, threshold2=100.0):
        # 拆分图像通道
        img_r = img[:,0:1] # red channel
        img_g = img[:,1:2] # green channel
        img_b = img[:,2:3] # blue channel

        # Step1: 应用高斯滤波进行模糊降噪
        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_g)
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_b)
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

        # Step2: 用 Sobel 算子求图像的强度梯度
        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)

        # Step2: 确定边缘梯度和方向
        grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2)
        grad_mag += torch.sqrt(grad_x_g**2 + grad_y_g**2)
        grad_mag += torch.sqrt(grad_x_b**2 + grad_y_b**2)
        grad_orientation = (torch.atan2(grad_y_r+grad_y_g+grad_y_b, grad_x_r+grad_x_g+grad_x_b) * (180.0/math.pi))
        grad_orientation += 180.0
        grad_orientation =  torch.round(grad_orientation / 45.0) * 45.0

        # Step3: 非最大抑制，边缘细化
        all_filtered = self.directional_filter(grad_mag)

        inidices_positive = (grad_orientation / 45) % 8
        inidices_negative = ((grad_orientation / 45) + 4) % 8

        batch, _, height, width = inidices_positive.shape
        pixel_count = height * width * batch
        pixel_range = torch.Tensor([range(pixel_count)]).to(self.device)

        indices = (inidices_positive.reshape((-1, )) * pixel_count + pixel_range).squeeze()
        channel_select_filtered_positive = all_filtered.reshape((-1, ))[indices.long()].reshape((batch, 1, height, width))

        indices = (inidices_negative.reshape((-1, )) * pixel_count + pixel_range).squeeze()
        channel_select_filtered_negative = all_filtered.reshape((-1, ))[indices.long()].reshape((batch, 1, height, width))

        channel_select_filtered = torch.stack([channel_select_filtered_positive, channel_select_filtered_negative])

        is_max = channel_select_filtered.min(dim=0)[0] > 0.0

        thin_edges = grad_mag.clone()
        thin_edges[is_max==0] = 0.0

        # Step4: 双阈值
        low_threshold = min(threshold1, threshold2)
        high_threshold = max(threshold1, threshold2)
        thresholded = thin_edges.clone()
        lower = thin_edges<low_threshold
        thresholded[lower] = 0.0
        higher = thin_edges>high_threshold
        thresholded[higher] = 1.0
        connect_map = self.connect_filter(higher.float())
        middle = torch.logical_and(thin_edges>=low_threshold, thin_edges<=high_threshold)
        thresholded[middle] = 0.0
        connect_map[torch.logical_not(middle)] = 0
        thresholded[connect_map>0] = 1.0
        thresholded[..., 0, :] = 0.0
        thresholded[..., -1, :] = 0.0
        thresholded[..., :, 0] = 0.0
        thresholded[..., :, -1] = 0.0
        thresholded = (thresholded>0.0).float()

        # return torch.add(torch.mul(self.w, torch.div(thresholded, torch.Tensor([255.]).to(self.device))), self.b)
        if self.detach:
            return torch.div(thresholded, torch.Tensor([255.]).to(self.device))
        else:
            return torch.div(torch.add(torch.mul(self.w, thresholded), self.b), torch.Tensor([255.]).to(self.device))