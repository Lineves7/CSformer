
import collections
import logging
import math
import os
import time
from datetime import datetime

import dateutil.tz
import torch
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
import cv2


def create_logger(log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def set_log_dir(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    # set log path
    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    # prefix = exp_path + '_' + timestamp
    prefix = exp_path
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    path_dict['prefix'] = prefix

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    path_dict['log_path'] = log_path

    # set sample image path for fid calculation
    sample_path = os.path.join(prefix, 'Samples')
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    path_dict['sample_path'] = sample_path

    return path_dict


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))


class RunningStats:
    def __init__(self, WIN_SIZE):
        self.mean = 0
        self.run_var = 0
        self.WIN_SIZE = WIN_SIZE

        self.window = collections.deque(maxlen=WIN_SIZE)

    def clear(self):
        self.window.clear()
        self.mean = 0
        self.run_var = 0

    def is_full(self):
        return len(self.window) == self.WIN_SIZE

    def push(self, x):

        if len(self.window) == self.WIN_SIZE:
            # Adjusting variance
            x_removed = self.window.popleft()
            self.window.append(x)
            old_m = self.mean
            self.mean += (x - x_removed) / self.WIN_SIZE
            self.run_var += (x + x_removed - old_m - self.mean) * (x - x_removed)
        else:
            # Calculating first variance
            self.window.append(x)
            delta = x - self.mean
            self.mean += delta / len(self.window)
            self.run_var += delta * (x - self.mean)

    def get_mean(self):
        return self.mean if len(self.window) else 0.0

    def get_var(self):
        return self.run_var / len(self.window) if len(self.window) > 1 else 0.0

    def get_std(self):
        return math.sqrt(self.get_var())

    def get_all(self):
        return list(self.window)

    def __str__(self):
        return "Current window values: {}".format(list(self.window))

def imread_CS_py(imgName):
    block_size = 64
    # img = cv2.imread(imgName)
    # img = cv2.cvtColor(imgName, cv2.COLOR_BGR2RGB)
    # Iorg = np.array(img, dtype='float32')
    Iorg = np.array(Image.open(imgName), dtype='float32')  # 读图
    if len(Iorg.shape) == 3: #rgb转y
        Iorg = test_rgb2ycbcr(Iorg)
    [row, col] = Iorg.shape  # 图像的 形状
    row_pad = block_size-np.mod(row,block_size)  # 求余数操作
    col_pad = block_size-np.mod(col,block_size)  # 求余数操作，用于判断需要补零的数量
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col+col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]


def imread_CS_py_new(imgName,block_size = 8):
    Iorg = np.array(Image.open(imgName), dtype='float32')  # 读图
    if len(Iorg.shape) == 3: #rgb转y
        Iorg = test_rgb2ycbcr(Iorg)
    # [row, col] = Iorg.shape  # 图像的 形状
    # row_pad = block_size-np.mod(row,block_size)  # 求余数操作
    # col_pad = block_size-np.mod(col,block_size)  # 求余数操作，用于判断需要补零的数量
    # Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    # Ipad = np.concatenate((Ipad, np.zeros([row_pad, col+col_pad])), axis=0)
    # [row_new, col_new] = Ipad.shape

    return Iorg

def img2col_py(Ipad, block_size):
    [row, col] = Ipad.shape  # 当前图像的形状
    row_block = row/block_size
    col_block = col/block_size
    block_num = int(row_block*col_block)  # 一共有多少个 模块
    img_col = np.zeros([block_size**2, block_num])  # 把每一块放进每一列中， 这就是容器
    count = 0
    for x in range(0, row-block_size+1, block_size):
        for y in range(0, col-block_size+1, block_size):
            img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].reshape([-1])
            # img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].transpose().reshape([-1])
            count = count + 1
    return img_col

def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def col2im_CS_py(X_col, row, col, row_new, col_new):
    block_size = 64
    X0_rec = np.zeros([row_new, col_new])
    count = 0
    for x in range(0, row_new-block_size+1, block_size):
        for y in range(0, col_new-block_size+1, block_size):
            X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size])
            # X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size]).transpose()
            count = count + 1
    X_rec = X0_rec[:row, :col]
    return X_rec

def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):

    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2

    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))

    return np.mean(np.mean(ssim_map))

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def batch_rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        img type:tensor
            size: [batch,channels,h,w] [5,3,64,64]
            device: gpu
            range: [-1,1]
        uint8, [0, 255]
        float, [0, 1]
    '''
    device = img.get_device()
    # img = img.to('cpu').numpy()
    img = (img + 1.) * 127.5
    img = img.permute((0,2,3,1)) #[batch,h,w,channels]
    # convert
    w = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)
    rlt = torch.matmul(img, w) / 255.0 + 16.0
    rlt = rlt/127.5 - 1. #[batch,h,w]
    rlt = torch.unsqueeze(rlt,1) #[batch,1,h,w]

    return rlt

def test_rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]

    rlt = rlt.round()

    return rlt.astype(in_img_type)


def img2patches(imgs,patch_size:tuple,stride_size:tuple):
    """
    Args:
        imgs: (H,W)/(H,W,C)/(B,H,W,C)
        patch_size: (patch_h, patch_w)
        stride_size: (stride_h, stride_w)
    """


    if imgs.ndim == 2:
        # (H,W) -> (1,H,W,1)
        imgs = np.expand_dims(imgs,axis=2)
        imgs = np.expand_dims(imgs,axis=0)
    elif imgs.ndim == 3:
        # (H,W,C) -> (1,H,W,C)
        imgs = np.expand_dims(imgs,axis=0)
    b,h,w,c = imgs.shape
    p_h,p_w = patch_size
    s_h,s_w = stride_size

    assert (h-p_h) % s_h == 0 and (w-p_w) % s_w == 0

    n_patches_y = (h - p_h) // s_h + 1
    n_patches_x = (w - p_w) // s_w + 1
    n_patches_per_img = n_patches_y * n_patches_x
    n_patches = n_patches_per_img * b
    patches = np.empty((n_patches,p_h,p_w,c),dtype=imgs.dtype)

    patch_idx = 0
    for img in imgs:
        for i in range(n_patches_y):
            for j in range(n_patches_x):
                y1 = i * s_h
                y2 = y1 + p_h
                x1 = j * s_w
                x2 = x1 + p_w
                patches[patch_idx] = img[y1:y2, x1:x2]
                patch_idx += 1
    return patches

def unpatch2d(patches, imsize: tuple, stride_size: tuple):
    '''
        patches: (n_patches, p_h, p_w,c)
        imsize: (img_h, img_w)
    '''
    assert len(patches.shape) == 4

    i_h, i_w = imsize
    n_patches,p_h,p_w,c = patches.shape
    s_h, s_w = stride_size

    assert (i_h - p_h) % s_h == 0 and (i_w - p_w) % s_w == 0

    n_patches_y = (i_h - p_h) // s_h + 1
    n_patches_x = (i_w - p_w) // s_w + 1
    n_patches_per_img = n_patches_y * n_patches_x
    batch_size = n_patches // n_patches_per_img

    imgs = np.zeros((batch_size,i_h,i_w,c))
    weights = np.zeros_like(imgs)


    for img_idx, (img,weights) in enumerate(zip(imgs,weights)):
        start = img_idx * n_patches_per_img

        for i in range(n_patches_y):
            for j in range(n_patches_x):
                y1 = i * s_h
                y2 = y1 + p_h
                x1 = j * s_w
                x2 = x1 + p_w
                patch_idx = start + i*n_patches_x+j
                img[y1:y2,x1:x2] += patches[patch_idx]
                weights[y1:y2, x1:x2] += 1
    imgs /= weights

    return imgs.astype(patches.dtype)
