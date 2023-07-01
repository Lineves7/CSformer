import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import functools





def pixel_upsample(x, H, W):
    B, N, C = x.size()
    assert N == H*W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H*W)
    x = x.permute(0,2,1)
    return x, H, W

class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel,up_mode='bicubic'):
        super(Upsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
        )
        self.up_mode = up_mode

    def forward(self, x):
        #up
        x = F.interpolate(x, scale_factor=2, mode=self.up_mode)
        #↓dim
        out = self.conv(x) # B H*W C
        return out

class ResGenerator(nn.Module):
    def __init__(self, args,upsample=Upsample):
        super(ResGenerator, self).__init__()
        self.args = args
        self.bottom_width = args.bottom_width
        self.embed_dim = conv_dim = args.gf_dim
        self.dec1 = nn.Sequential(
            ResBlock(in_channels=conv_dim, out_channels=conv_dim,norm_fun = args.cnnnorm_type),
            ResBlock(in_channels=conv_dim, out_channels=conv_dim,norm_fun = args.cnnnorm_type))# 8*8*128 --> 32*32*256
        self.upsample_1 = upsample(conv_dim, conv_dim // 2)
        self.dec2 = nn.Sequential(
            ResBlock(in_channels=conv_dim// 2, out_channels=conv_dim// 2,norm_fun = args.cnnnorm_type),
            ResBlock(in_channels=conv_dim// 2, out_channels=conv_dim// 2,norm_fun = args.cnnnorm_type))  # 16*16*128 --> 32*32*256
        self.upsample_2 = upsample(conv_dim//2, conv_dim // 4)
        self.dec3 =nn.Sequential(
            ResBlock(in_channels=conv_dim// 4, out_channels=conv_dim// 4,norm_fun = args.cnnnorm_type),
            ResBlock(in_channels=conv_dim// 4, out_channels=conv_dim// 4,norm_fun = args.cnnnorm_type))  # 32*32*128 --> 32*32*256
        self.upsample_3 = upsample(conv_dim // 4, conv_dim // 8)
        self.dec4 = nn.Sequential(
            ResBlock(in_channels=conv_dim// 8, out_channels=conv_dim// 8,norm_fun = args.cnnnorm_type),
            ResBlock(in_channels=conv_dim// 8, out_channels=conv_dim// 8,norm_fun = args.cnnnorm_type))  # 64*64*128 --> 32*32*256
        # self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * self.embed_dim)

        self.to_rgb = nn.ModuleList()
        self.padding3 = (3 + (3 - 1) * (1 - 1) - 1) // 2
        self.padding7 = (7 + (7 - 1) * (1 - 1) - 1) // 2
        # for i in range(3):
        #     to_rgb = None
        #     dim = conv_dim // (2**i)
        #     to_rgb = nn.Sequential(
        #         nn.ReflectionPad2d(self.padding3),
        #         nn.Conv2d(dim, 1, 3, 1, 0),
        #         # nn.ReflectionPad2d(self.padding7),
        #         # nn.Conv2d(conv_dim, 1, 7, 1, 0),
        #         nn.Tanh()
        #     )
        #     self.to_rgb.append(to_rgb)

    def forward(self, x):
        features = []
        rgb = []
        x = x.permute(0,2,1).contiguous().view(-1,self.embed_dim, self.bottom_width,self.bottom_width)

        #8x8
        x = self.dec1(x)
        features.append(x)
        # rgb.append(self.to_rgb[0](x))

        #16x16
        x = self.upsample_1(x)
        x = self.dec2(x)
        features.append(x)
        # rgb.append(self.to_rgb[1](x))

        #32x32
        x = self.upsample_2(x)
        x = self.dec3(x)
        features.append(x)
        # rgb.append(self.to_rgb[2](x))

        #64x64
        x = self.upsample_3(x)
        x = self.dec4(x)
        features.append(x)

        return features,rgb


class Generator(nn.Module):
    def __init__(self, args,upsample=Upsample):
        super(Generator, self).__init__()
        self.args = args
        self.bottom_width = args.bottom_width
        self.embed_dim = conv_dim = args.gf_dim
        self.dec1 = ConvBlock(in_channels=conv_dim, out_channels=conv_dim,norm_fun = args.cnnnorm_type)  # 8*8*128 --> 32*32*256
        self.upsample_1 = upsample(conv_dim, conv_dim // 2)
        self.dec2 = ConvBlock(in_channels=conv_dim//2, out_channels=conv_dim//2,norm_fun = args.cnnnorm_type)  # 16*16*128 --> 32*32*256
        self.upsample_2 = upsample(conv_dim//2, conv_dim // 4)
        self.dec3 = ConvBlock(in_channels=conv_dim//4, out_channels=conv_dim//4,norm_fun = args.cnnnorm_type)  # 32*32*128 --> 32*32*256
        self.upsample_3 = upsample(conv_dim // 4, conv_dim // 8)
        self.dec4 = ConvBlock(in_channels=conv_dim//8, out_channels=conv_dim//8,norm_fun = args.cnnnorm_type)  # 64*64*128 --> 32*32*256
        # self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * self.embed_dim)

    def forward(self, x):
        features = []
        x = tf2cnn(x)

        #8x8
        x = self.dec1(x)
        features.append(x)


        #16x16
        x = self.upsample_1(x)
        x = self.dec2(x)
        features.append(x)


        #32x32
        x = self.upsample_2(x)
        x = self.dec3(x)
        features.append(x)

        #64x64
        x = self.upsample_3(x)
        x = self.dec4(x)
        features.append(x)

        return features


class Generator_nopos(nn.Module):
    def __init__(self, args,upsample=Upsample):
        super(Generator_nopos, self).__init__()
        self.args = args
        self.bottom_width = args.bottom_width
        self.embed_dim = conv_dim = args.gf_dim
        self.dec1 = ConvBlock(in_channels=conv_dim, out_channels=conv_dim,norm_fun = args.cnnnorm_type)  # 8*8*128 --> 32*32*256
        self.upsample_1 = upsample(conv_dim, conv_dim // 2)
        self.dec2 = ConvBlock(in_channels=conv_dim//2, out_channels=conv_dim//2,norm_fun = args.cnnnorm_type)  # 16*16*128 --> 32*32*256
        self.upsample_2 = upsample(conv_dim//2, conv_dim // 4)
        self.dec3 = ConvBlock(in_channels=conv_dim//4, out_channels=conv_dim//4,norm_fun = args.cnnnorm_type)  # 32*32*128 --> 32*32*256
        self.upsample_3 = upsample(conv_dim // 4, conv_dim // 8)
        self.dec4 = ConvBlock(in_channels=conv_dim//8, out_channels=conv_dim//8,norm_fun = args.cnnnorm_type)  # 64*64*128 --> 32*32*256
        # self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * self.embed_dim)

    def forward(self, x):
        features = []
        # x = tf2cnn(x)

        #8x8
        x = self.dec1(x)
        features.append(x)


        #16x16
        x = self.upsample_1(x)
        x = self.dec2(x)
        features.append(x)


        #32x32
        x = self.upsample_2(x)
        x = self.dec3(x)
        features.append(x)

        #64x64
        x = self.upsample_3(x)
        x = self.dec4(x)
        features.append(x)

        return features

def tf2cnn(x):
    B, L, C = x.shape
    H = int(math.sqrt(L))
    W = int(math.sqrt(L))
    x = x.transpose(1, 2).contiguous().view(B, C, H, W)
    return x

def cnn2tf(x):
    B,C,H,W = x.shape
    L = H*W
    x = x.flatten(2).transpose(1,2).contiguous()  # B H*W C
    return x,C,H,W

def pixel_upsample(x, H, W):
    B, N, C = x.size()
    assert N == H*W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H*W)
    x = x.permute(0,2,1)
    return x, H, W


def bicubic_upsample(x,H,W,up_mode='bicubic'):
    B, N, C = x.size()
    assert N == H*W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = F.interpolate(x, scale_factor=2, mode=up_mode)
    B, C, H, W = x.size()
    x = x.view(-1, C, H*W)
    x = x.permute(0,2,1)
    return x, H, W

def get_norm_fun(norm_fun_type='none'):
    if norm_fun_type == 'BatchNorm':
        norm_fun = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_fun_type == 'InstanceNorm':
        norm_fun = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=True)
    elif norm_fun_type == 'none':
        norm_fun = lambda x: Identity()
    else:
        raise NotImplementedError('normalization function [%s] is not found' % norm_fun_type)
    return norm_fun


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3,dilation = 1,norm_fun='none'):
        super(ConvBlock, self).__init__()
        self.padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        norm_fun = get_norm_fun(norm_fun)
        self.conv = nn.Sequential(
            #1
            nn.ReflectionPad2d(self.padding),
            nn.Conv2d(in_channels, out_channels, 3, 1, 0),
            norm_fun(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #2
            nn.ReflectionPad2d(self.padding),
            nn.Conv2d(in_channels, out_channels, 3, 1, 0),
            norm_fun(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3,dilation = 1,norm_fun='none'):
        super(ResBlock, self).__init__()
        self.padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        norm_fun = get_norm_fun(norm_fun)
        self.conv = nn.Sequential(
            #1
            nn.ReflectionPad2d(self.padding),
            nn.Conv2d(in_channels, out_channels, 3, 1, 0),
            norm_fun(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #2
            nn.ReflectionPad2d(self.padding),
            nn.Conv2d(in_channels, out_channels, 3, 1, 0),
            norm_fun(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x) + x


def get_act_fun(act_fun_type='LeakyReLU'):
    if isinstance(act_fun_type, str):
        if act_fun_type == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun_type == 'ReLU':
            return nn.ReLU(inplace=True)
        elif act_fun_type == 'SELU':
            return nn.SELU(inplace=True)
        elif act_fun_type == 'none':
            return nn.Sequential()
        else:
            raise NotImplementedError('activation function [%s] is not found' % act_fun_type)
    else:
        return act_fun_type()

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_fun(norm_fun_type='none'):
    if norm_fun_type == 'BatchNorm':
        norm_fun = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_fun_type == 'InstanceNorm':
        norm_fun = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=True)
    elif norm_fun_type == 'none':
        norm_fun = lambda x: Identity()
    else:
        raise NotImplementedError('normalization function [%s] is not found' % norm_fun_type)
    return norm_fun






