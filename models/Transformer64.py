import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from models.ViT_helper import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat


class matmul(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x1, x2):
        x = x1@x2
        return x

def count_matmul(m, x, y):
    num_mul = x[0].numel() * x[1].size(-1)
    # m.total_ops += torch.DoubleTensor([int(num_mul)])
    m.total_ops += torch.DoubleTensor([int(0)])
    

def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initialy created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def leakyrelu(x):
    return nn.functional.leaky_relu_(x, 0.2)
class CustomAct(nn.Module):
    def __init__(self, act_layer):
        super().__init__()
        if act_layer == "gelu":
            self.act_layer = gelu
        elif act_layer == "leakyrelu":
            self.act_layer = leakyrelu

    def forward(self, x):
        return self.act_layer(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=gelu, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = CustomAct(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def get_attn_mask(N, w):
    mask = torch.zeros(1, 1, N, N).cuda()
    for i in range(N):
        if i <= w:
            mask[:, :, i, 0:i+w+1] = 1
        elif N - i <= w:
            mask[:, :, i, i-w:N] = 1
        else:
            mask[:, :, i, i:i+w+1] = 1
            mask[:, :, i, i-w:i] = 1
    return mask


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=16):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mat = matmul()
        self.window_size = window_size

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (self.mat(q, k.transpose(-2, -1))) * self.scale
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = self.mat(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

#########################################
########### window operation#############
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PixelNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=2, keepdim=True) + 1e-8)

class CustomNorm(nn.Module):
    def __init__(self, norm_layer, dim):
        super().__init__()
        self.norm_type = norm_layer
        if norm_layer == "ln":
            self.norm = nn.LayerNorm(dim)
        elif norm_layer == "bn":
            self.norm = nn.BatchNorm1d(dim)
        elif norm_layer == "in":
            self.norm = nn.InstanceNorm1d(dim)
        elif norm_layer == "pn":
            self.norm = PixelNorm(dim)

    def forward(self, x):
        if self.norm_type == "bn" or self.norm_type == "in":
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            return x
        elif self.norm_type == "none":
            return x
        else:
            return self.norm(x)

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=gelu, norm_layer=nn.LayerNorm, window_size=16):
        super().__init__()
        self.norm1 = CustomNorm(norm_layer, dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, window_size=window_size)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = CustomNorm(norm_layer, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class StageBlock(nn.Module):
    def __init__(self, depth, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=gelu, norm_layer=nn.LayerNorm, window_size=16):
        super().__init__()
        self.depth = depth
        models = [Block(
                        dim=dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        window_size=window_size
                        ) for i in range(depth)]
        self.block = nn.Sequential(*models)
    def forward(self, x):
#         for blk in self.block:
#             # x = blk(x)
#             checkpoint.checkpoint(blk, x)
#         x = checkpoint.checkpoint(self.block, x)
        x = self.block(x)
        return x

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
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        #up
        x = F.interpolate(x, scale_factor=2, mode=self.up_mode)
        H = x.shape[2]
        W = x.shape[3]
        #↓dim
        out = self.conv(x).flatten(2).transpose(1,2).contiguous() # B H*W C
        return out,H,W

# --- Build dense --- #
class MakeDense(nn.Module):
    def __init__(self, in_channels, growth_rate, num_heads,mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=gelu, norm_layer=nn.LayerNorm, window_size=16):
        super(MakeDense, self).__init__()
        self.block =Block(
                        dim=in_channels,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        window_size=window_size
                        )
        self.conv_1x1 = nn.Conv2d(in_channels, growth_rate, kernel_size=1, padding=0)
        self.norm = CustomNorm('ln', growth_rate)

    def forward(self, x):
        out = self.block(x)

        B, N, C = out.size()
        H = W = int(math.sqrt(N))
        out = out.permute(0, 2, 1).contiguous()
        out = out.view(-1, C, H, W)
        out = self.conv_1x1(out)
        B, C, H, W = out.size()
        out = out.view(-1, C, H * W)
        out = out.permute(0, 2, 1).contiguous()
        out = self.norm(out)

        out = torch.cat((x, out), 2)

        return out


# --- Build the Residual Dense Block --- #
class RDTB(nn.Module):
    def __init__(self, num_dense_layer,in_channels,growth_rate, num_heads,mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=gelu, norm_layer=nn.LayerNorm, window_size=16):
        """
        :param in_channels: input channel size
        :param num_dense_layer: the number of RDB layers
        :param growth_rate: growth_rate
        """
        super(RDTB, self).__init__()
        _in_channels = in_channels
        modules = []
        for i in range(num_dense_layer):
            modules.append(MakeDense(in_channels=_in_channels,
                                     growth_rate = growth_rate,
                                     num_heads=num_heads,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     drop=drop,
                                     attn_drop=attn_drop,
                                     drop_path=drop_path,
                                     act_layer=act_layer,
                                     norm_layer=norm_layer,
                                     window_size=window_size
                        ))
            _in_channels += growth_rate
        self.residual_dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)
        self.norm = CustomNorm('ln', in_channels)

    def forward(self, x):
        out = self.residual_dense_layers(x)

        B, N, C = out.size()
        H = W = int(math.sqrt(N))
        out = out.permute(0, 2, 1).contiguous()
        out = out.view(-1, C, H, W)
        out = self.conv_1x1(out)
        B, C, H, W = out.size()
        out = out.view(-1, C, H * W)
        out = out.permute(0, 2, 1).contiguous()
        out = self.norm(out)
        out = out + x
        return out

class Generator_tailadd(nn.Module):
    def __init__(self, args, img_size=224, patch_size=16, in_chans=3, num_classes=10, embed_dim=384, depth=[2,2,2,2],
                 num_heads=[8,4,2,1], mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm,upsample=Upsample):
        super().__init__()
        self.args = args
        growth_rate = args.growth_rate
        self.ch = embed_dim
        self.bottom_width = args.bottom_width
        self.embed_dim = embed_dim = args.gf_dim
        self.window_size = args.g_window_size
        num_dense_layer = args.num_dense_layer
        norm_layer = args.g_norm
        mlp_ratio = args.g_mlp
        depth = [int(i) for i in args.g_depth.split(",")]
        act_layer = args.g_act
        num_heads = [int(i) for i in args.num_heads.split(",")]

        self.pos_embed_1 = nn.Parameter(torch.zeros(1, self.bottom_width ** 2, embed_dim))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, (self.bottom_width * 2) ** 2, embed_dim//2))
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, (self.bottom_width * 4) ** 2, embed_dim//4))
        self.pos_embed_4 = nn.Parameter(torch.zeros(1, (self.bottom_width * 8) ** 2, embed_dim//8))

        self.pos_embed = [
            self.pos_embed_1,
            self.pos_embed_2,
            self.pos_embed_3,
            self.pos_embed_4
        ]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth[0])]  # stochastic depth decay rule

        self.blocks_1 = StageBlock(
                        depth=depth[0],
                        dim=embed_dim,
                        num_heads=num_heads[0],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=0,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        window_size=self.window_size
                        )
        self.upsample_1 = upsample(embed_dim, embed_dim//2)

        self.blocks_2 = StageBlock(
                        depth=depth[1],
                        dim=embed_dim//2,
                        num_heads=num_heads[1],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=0,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        window_size=self.window_size
                        )
        self.upsample_2 = upsample(embed_dim//2, embed_dim // 4)

        self.blocks_3 = StageBlock(
                        depth=depth[2],
                        dim=embed_dim//4,
                        num_heads=num_heads[2],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=0,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        window_size=self.window_size
                        )
        self.upsample_3 = upsample(embed_dim // 4, embed_dim // 8)

        self.blocks_4 = StageBlock(
                        depth=depth[3],
                        dim=embed_dim//8,
                        num_heads=num_heads[3],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=0,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        window_size=self.window_size
                        )


        for i in range(len(self.pos_embed)):
            trunc_normal_(self.pos_embed[i], std=.02)


        self.padding3 = (3 + (3 - 1) * (1 - 1) - 1) // 2
        self.padding7 = (7 + (7 - 1) * (1 - 1) - 1) // 2
        self.to_rgb = nn.ModuleList()
        for i in range(4):
            to_rgb = None
            dim = embed_dim // (2**i)
            to_rgb = nn.Sequential(
                nn.ReflectionPad2d(self.padding3),
                nn.Conv2d(dim, 1, 3, 1, 0),
                # nn.ReflectionPad2d(self.padding7),
                # nn.Conv2d(self.embed_dim, 1, 7, 1, 0),
                nn.Tanh()
            )
            self.to_rgb.append(to_rgb)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x, gsfeatures, gsrgb):

        features = []
        fufeatures = []
        tfrgb = []
        outputrgb = []
        self.pos_embed = self.pos_embed
        #change

        #-------block 1----------
        x = x + self.pos_embed[0].to(x.get_device())  #8x8
        B,_,C = x.size()
        H, W = self.bottom_width, self.bottom_width
        x = self.blocks_1(x)
        #features add
        features.append(x.permute(0, 2, 1).contiguous().view(-1, C, H, W))
        tfrgb.append(self.to_rgb[0](x.permute(0, 2, 1).contiguous().view(-1, C, H, W)))

        fu_rgb = tfrgb[0]+gsrgb[0]
        fu_feature = features[0] + gsfeatures[0]
        outputrgb.append(fu_rgb)
        fufeatures.append(fu_feature)
        x = fu_feature.view(-1,C,H*W).permute(0,2,1).contiguous()

        # -------block 2----------
        x, H, W = self.upsample_1(x) #16x16
        x = x + self.pos_embed[1].to(x.get_device())
        B, _, C = x.size()
        x = x.view(B, H, W, C)
        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size * self.window_size, C)
        x = self.blocks_2(x)
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, H, W).view(B, H * W, C)
        #features add
        features.append(x.permute(0, 2, 1).contiguous().view(-1, C, H, W))
        tfrgb.append(self.to_rgb[1](x.permute(0, 2, 1).contiguous().view(-1, C, H, W)))

        fu_rgb = tfrgb[1]+gsrgb[1]
        fu_feature = features[1] + gsfeatures[1]
        outputrgb.append(fu_rgb)
        fufeatures.append(fu_feature)
        x = fu_feature.view(-1,C,H*W).permute(0,2,1).contiguous()

        # -------block 3----------
        x, H, W = self.upsample_2(x) #32x32
        x = x + self.pos_embed[2].to(x.get_device())
        B, _, C = x.size()
        x = x.view(B, H, W, C)
        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size * self.window_size, C)
        x = self.blocks_3(x)
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, H, W).view(B, H * W, C)
        #features add
        features.append(x.permute(0, 2, 1).contiguous().view(-1, C, H, W))
        tfrgb.append(self.to_rgb[2](x.permute(0, 2, 1).contiguous().view(-1, C, H, W)))

        fu_rgb = tfrgb[2]+gsrgb[2]
        fu_feature = features[2] + gsfeatures[2]
        outputrgb.append(fu_rgb)
        fufeatures.append(fu_feature)
        x = fu_feature.view(-1,C,H*W).permute(0,2,1).contiguous()

        # -------block 4----------
        x, H, W = self.upsample_3(x) #64x64
        x = x + self.pos_embed[3].to(x.get_device())
        B, _, C = x.size()
        x = x.view(B, H, W, C)
        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size * self.window_size, C)
        x = self.blocks_4(x)
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, H, W).view(B, H * W, C)
        #features add
        features.append(x.permute(0, 2, 1).contiguous().view(-1, C, H, W))


        fu_feature = features[3] + gsfeatures[3]
        fufeatures.append(fu_feature)
        output = self.to_rgb[3](fu_feature)
        outputrgb.append(output)

        final_output = outputrgb.pop(-1)
        return final_output, outputrgb


class Transformer(nn.Module):
    def __init__(self, args, img_size=224, patch_size=16, in_chans=3, num_classes=10, embed_dim=384, depth=[2,2,2,2],
                 num_heads=[16,8,4,2], mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm,upsample=Upsample):
        super().__init__()
        self.args = args
        self.ch = embed_dim
        self.bottom_width = args.bottom_width
        self.embed_dim = embed_dim = args.gf_dim
        self.window_size = args.g_window_size
        norm_layer = args.g_norm
        mlp_ratio = args.g_mlp
        depth = [int(i) for i in args.g_depth.split(",")]
        act_layer = args.g_act
        num_heads = [int(i) for i in args.num_heads.split(",")]

        self.pos_embed_1 = nn.Parameter(torch.zeros(1, self.bottom_width ** 2, embed_dim*2))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, (self.bottom_width * 2) ** 2, embed_dim))
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, (self.bottom_width * 4) ** 2, embed_dim//2))
        self.pos_embed_4 = nn.Parameter(torch.zeros(1, (self.bottom_width * 8) ** 2, embed_dim//4))

        self.pos_embed = [
            self.pos_embed_1,
            self.pos_embed_2,
            self.pos_embed_3,
            self.pos_embed_4
        ]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth[0])]  # stochastic depth decay rule

        self.blocks_1 = StageBlock(
                        depth=depth[0],
                        dim=embed_dim*2,
                        num_heads=num_heads[0],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=0,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        window_size=self.window_size
                        )
        # self.upsample_1 = upsample(embed_dim, embed_dim//2)

        self.blocks_2 = StageBlock(
                        depth=depth[1],
                        dim=embed_dim,
                        num_heads=num_heads[1],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=0,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        window_size=self.window_size
                        )
        # self.upsample_2 = upsample(embed_dim//2, embed_dim // 4)

        self.blocks_3 = StageBlock(
                        depth=depth[2],
                        dim=embed_dim//2,
                        num_heads=num_heads[2],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=0,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        window_size=self.window_size
                        )
        # self.upsample_3 = upsample(embed_dim // 4, embed_dim // 8)

        self.blocks_4 = StageBlock(
                        depth=depth[3],
                        dim=embed_dim//4,
                        num_heads=num_heads[3],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=0,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        window_size=self.window_size
                        )

        for i in range(len(self.pos_embed)):
            trunc_normal_(self.pos_embed[i], std=.02)

        if args.datarange == '01':
            rgbact = nn.Sigmoid()
            print('You choose sigmoid for range [0,1]')
        elif args.datarange == '-11':
            rgbact = nn.Tanh()
            print('You choose tanh for range [-1,1]')

        self.padding3 = (3 + (3 - 1) * (1 - 1) - 1) // 2
        self.padding7 = (7 + (7 - 1) * (1 - 1) - 1) // 2
        self.to_rgb = nn.Sequential(
                nn.ReflectionPad2d(self.padding3),
                nn.Conv2d((embed_dim*2) // (2**3), (embed_dim*2) // (2**3), 3, 1, 0),
                nn.ReflectionPad2d(self.padding7),
                nn.Conv2d((embed_dim*2) // (2**3), 1, 7, 1, 0),
                nn.Tanh()
            )

        self.apply(self._init_weights)
        print('apply init weith trunc_normal')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, gsfeatures, inirgb):

        features = []
        # fufeatures = []
        # tfrgb = []
        outputrgb = []

        #change

        #-------block 1----------
        #feature cat
        x = tf2cnn(x)
        x = torch.cat([gsfeatures[0],x],1)
        x,C,H,W = cnn2tf(x)
        x = x + self.pos_embed[0].to(x.get_device())
        x = self.blocks_1(x)


        # -------block 2----------
        x, H, W = pixel_upsample(x,H,W) #16x16
        x = tf2cnn(x)
        x = torch.cat([gsfeatures[1],x],1)
        x,C,H,W = cnn2tf(x)
        x = x + self.pos_embed[1].to(x.get_device())
        B, _, C = x.size()
        x = x.view(B, H, W, C)
        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size * self.window_size, C)
        x = self.blocks_2(x)
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, H, W).view(B, H * W, C)



        # -------block 3----------
        x, H, W = pixel_upsample(x,H,W) #32x32
        x = tf2cnn(x)
        x = torch.cat([gsfeatures[2],x],1)
        x,C,H,W = cnn2tf(x)
        x = x + self.pos_embed[2].to(x.get_device())
        B, _, C = x.size()
        x = x.view(B, H, W, C)
        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size * self.window_size, C)
        x = self.blocks_3(x)
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, H, W).view(B, H * W, C)


        # -------block 4----------
        x, H, W = pixel_upsample(x,H,W) #64x64
        x = tf2cnn(x)
        x = torch.cat([gsfeatures[3],x],1)
        x,C,H,W = cnn2tf(x)
        x = x + self.pos_embed[3].to(x.get_device())
        B, _, C = x.size()
        x = x.view(B, H, W, C)
        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size * self.window_size, C)
        x = self.blocks_4(x)
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, H, W).view(B, H * W, C)
        rgb_64 =self.to_rgb(x.permute(0, 2, 1).view(-1, C, H, W))+inirgb

        return rgb_64

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



def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


def get_activation(activation):
    """
    Get the module for a specific activation function and its gain if
    it can be calculated.
    Arguments:
        activation (str, callable, nn.Module): String representing the activation.
    Returns:
        activation_module (torch.nn.Module): The module representing
            the activation function.
        gain (float): The gain value. Defaults to 1 if it can not be calculated.
    """
    if isinstance(activation, nn.Module) or callable(activation):
        return activation
    if isinstance(activation, str):
        activation = activation.lower()
    if activation in [None, 'linear']:
        return nn.Identity()
    lrelu_strings = ('leaky', 'leakyrely', 'leaky_relu', 'leaky relu', 'lrelu')
    if activation.startswith(lrelu_strings):
        for l_s in lrelu_strings:
            activation = activation.replace(l_s, '')
        slope = ''.join(
            char for char in activation if char.isdigit() or char == '.')
        slope = float(slope) if slope else 0.01
        return nn.LeakyReLU(slope)  # close enough to true gain
    elif activation in ['relu']:
        return nn.ReLU()
    elif activation in ['elu']:
        return nn.ELU()
    elif activation in ['prelu']:
        return nn.PReLU()
    elif activation in ['rrelu', 'randomrelu']:
        return nn.RReLU()
    elif activation in ['selu']:
        return nn.SELU()
    elif activation in ['softplus']:
        return nn.Softplus()
    elif activation in ['softsign']:
        return nn.Softsign()  # unsure about this gain
    elif activation in ['sigmoid', 'logistic']:
        return nn.Sigmoid()
    elif activation in ['tanh']:
        return nn.Tanh()
    else:
        raise ValueError(
            'Activation "{}" not available.'.format(activation)
        )


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict



