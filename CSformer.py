import torch
import torch.nn as nn
import torch.nn.functional as F
import models.Transformer64 as Transformer64
import models.CNN64 as CNN64

import numpy as np

class CSformer(nn.Module):
    def __init__(self, args):
        super(CSformer, self).__init__()

        self.args = args
        ## sampling matrix
        # self.register_parameter("sensing_matrix", nn.Parameter(torch.from_numpy(sensing_matrix).float(), requires_grad=True))
        self.n_input = args.n_input
        self.bottom_width = args.bottom_width
        self.embed_dim = args.gf_dim
        # self.l1 = nn.Linear(args.latent_dim, (args.bottom_width ** 2) * args.gf_dim)
        self.outdim = int(np.ceil((args.img_size**2)//(args.bottom_width ** 2)))
        # self.outdim = int(np.ceil(args.latent_dim // (args.bottom_width ** 2)))
        self.iniconv = nn.Sequential(
            # nn.ReflectionPad2d(1),
            nn.Conv2d(self.n_input,128,1,1,0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 1, 1, 0)
        )
        self.act = nn.LeakyReLU(0.2,inplace=True)

        self.Phi = nn.Parameter(torch.nn.init.xavier_normal_(torch.Tensor(self.n_input, 256)))
        self.PhiT = nn.Parameter(torch.nn.init.xavier_normal_(torch.Tensor(256,self.n_input)))
        # self.nsv = args.nsv

        #transformer branch
        self.td = Transformer64.Transformer(args)


        #cnn branch
        self.gs = CNN64.Generator(args)




    def together(self,inputs,S,H,L):
        inputs = inputs.squeeze(1)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=H*S, dim=0), dim=2)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=S, dim=0), dim=1)
        inputs = inputs.unsqueeze(1)
        return inputs

    def forward(self, inputs):

        H = int(inputs.shape[2]/64)
        L = int(inputs.shape[3]/64)
        S = inputs.shape[0]
        inputs = torch.squeeze(inputs,dim=1)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=64, dim=1), dim=0)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=64, dim=2), dim=0)
        inputs = torch.unsqueeze(inputs,dim=1)

        np.random.seed(12345)
        PhiWeight = self.Phi.contiguous().view(self.n_input, 1, 16, 16)
        y = F.conv2d(inputs, PhiWeight, padding=0, stride=16, bias=None)


        # Initialization-subnet
        PhiTWeight = self.PhiT.contiguous().view(256, self.n_input, 1, 1)
        PhiTb = F.conv2d(y, PhiTWeight, padding=0, bias=None)
        PhiTb = torch.nn.PixelShuffle(16)(PhiTb)

        x = self.iniconv(y)
        x = torch.nn.PixelShuffle(2)(x)

        x  =x.flatten(2).transpose(1,2).contiguous()
        gsfeatures = self.gs(x)
        output = self.td(x,gsfeatures,PhiTb)
        merge_output = self.together(output, S, H, L)
        merge_PhiTb = self.together(PhiTb, S, H, L)
        # final_output = self.enhance(merge_output,merge_PhiTb)


        return merge_output, merge_PhiTb, output, PhiTb
        # return merge_output, merge_PhiTb, output, a

