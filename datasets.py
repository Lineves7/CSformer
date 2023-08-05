import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import random
from scipy import io
import cv2
import numpy as np

class Augment_RGB_torch:
    def __init__(self):
        pass
    def transform0(self, torch_tensor):
        return torch_tensor
    def transform1(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1,-2])
        return torch_tensor
    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1,-2])
        return torch_tensor
    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1,-2])
        return torch_tensor
    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor
    def transform5(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1,-2])).flip(-2)
        return torch_tensor
augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')]

class ImageDataset(object):
    def __init__(self, args, cur_img_size=None, bs=None):
        bs = args.gen_batch_size if bs == None else bs
        img_size = args.img_size
        if args.dataset.lower() == 'coco' or args.dataset.lower() == 'div2k':
            Dt = ImgData(args)
            self.train = torch.utils.data.DataLoader(Dt,batch_size=args.gen_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        elif args.dataset.lower() == 'bsd400':
            Dt = ImgData_BSD400(args)
            self.train = torch.utils.data.DataLoader(Dt,batch_size=args.gen_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        else:
            raise NotImplementedError('Unknown dataset: {}'.format(args.dataset))


class ImgData_BSD400():
    """imagent"""
    def __init__(self, args,train=True):
        self.dataroot = args.data_path

        self.img_list = search(os.path.join(self.dataroot), "png")
        self.img_list = sorted(self.img_list)
        # print(self.img_list)
        self.train = train

        self.args = args
        self.len = len(self.img_list)
        print("data length:", len(self.img_list))

    def __len__(self):
        return len(self.img_list)

    def _get_index(self, idx):
        return idx % len(self.img_list)

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_lr = self.img_list[idx]

        #CV

        lr_cv = cv2.imread(f_lr)
        lr_cv = cv2.cvtColor(lr_cv, cv2.COLOR_BGR2RGB)
        lr = rgb2ycbcr(lr_cv)
        if len(lr.shape) == 2:
            lr = np.expand_dims(lr, axis=2)


        return lr

    def _np2Tensor(self, img, rgb_range):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = np_transpose.astype(np.float32)
        tensor = tensor * (rgb_range / 255)
        return tensor

    def __getitem__(self, idx):
        hr = self._load_file(idx % self.len)
        hr = self.get_patch(hr)
        hr = self._np2Tensor(hr, rgb_range=255)

        if self.args.datarange == '01':
            hr = hr/255.
        else:
            hr = hr / 127.5 - 1
        #aug
        apply_trans = transforms_aug[random.getrandbits(3)]
        hr = torch.from_numpy(hr)
        hr = getattr(augment, apply_trans)(hr)


        return hr

    def get_patch(self, lr, scale=1):
        lr = get_patch_img(lr, patch_size=self.args.train_patch_size, scale=scale)
        return lr

class ImgData():
    """imagent"""
    def __init__(self, args,train=True):
        self.dataroot = args.data_path
        if args.dataset.lower() == 'coco':
            self.img_list = search(os.path.join(self.dataroot), "jpg")
        else:
            self.img_list = search(os.path.join(self.dataroot), "png")
        self.img_list = sorted(self.img_list)[:(len(self.img_list)//3)]
        self.train = train

        self.args = args
        self.len = len(self.img_list)
        print("data length:", len(self.img_list))

    def __len__(self):
        return len(self.img_list)

    def _get_index(self, idx):
        return idx % len(self.img_list)

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_lr = self.img_list[idx]

        #CV

        lr_cv = cv2.imread(f_lr)
        lr_cv = cv2.cvtColor(lr_cv, cv2.COLOR_BGR2RGB)
        lr = rgb2ycbcr(lr_cv)
        if len(lr.shape) == 2:
            lr = np.expand_dims(lr, axis=2)

        return lr

    def _np2Tensor(self, img, rgb_range):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = np_transpose.astype(np.float32)
        tensor = tensor * (rgb_range / 255)
        return tensor

    def __getitem__(self, idx):
        hr = self._load_file(idx % self.len)
        hr = self.get_patch(hr)
        hr = self._np2Tensor(hr, rgb_range=255)

        if self.args.datarange == '01':
            hr = hr/255.
        else:
            hr = hr / 127.5 - 1
        #aug
        apply_trans = transforms_aug[random.getrandbits(3)]
        hr = torch.from_numpy(hr)
        hr = getattr(augment, apply_trans)(hr)


        return hr

    def get_patch(self, lr, scale=1):
        lr = get_patch_img(lr, patch_size=self.args.train_patch_size, scale=scale)
        return lr


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def search(root, target="JPEG"):
    """imagent"""
    item_list = []
    items = os.listdir(root)
    for item in items:
        path = os.path.join(root, item)
        if os.path.isdir(path):
            item_list.extend(search(path, target))
        elif path.split('.')[-1] == target:
            item_list.append(path)
        elif path.split('/')[-1].startswith(target):
            item_list.append(path)
    return item_list


def get_patch_img(img, patch_size=128, scale=1):
    """imagent"""
    ih, iw = img.shape[:2]
    tp = scale * patch_size
    if (iw - tp) > -1 and (ih-tp) > 1:
        ix = random.randrange(0, iw-tp+1)
        iy = random.randrange(0, ih-tp+1)
        hr = img[iy:iy+tp, ix:ix+tp, :]
    else:
        img = np.resize(img,(ih*2,iw*2,1))
        ih, iw = img.shape[:2]
        if (iw - tp) > -1 and (ih - tp) > 1:
            tp = scale * patch_size
            ix = random.randrange(0, iw - tp + 1)
            iy = random.randrange(0, ih - tp + 1)
            hr = img[iy:iy + tp, ix:ix + tp, :]
    return hr


