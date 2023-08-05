from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cfg
import torch
import os
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from functions import train, validate, load_params, copy_params
import datasets
from utils.utils import set_log_dir, save_checkpoint, create_logger
from tensorboardX import SummaryWriter
from loss.loss import get_loss_dict
import time


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)

    # Get Sensing_matrix
    # ratio_dict = {1: 10, 4: 43, 10: 103, 25: 272, 30: 327, 40: 436, 50: 545} #32*32
    ratio_dict = {1: 3, 4: 11, 10: 26, 25: 64, 30: 77, 40: 103, 50: 128} #16*16
    n_input = ratio_dict[args.cs_ratio]
    args.n_input = n_input

    # import network
    exec('import '+'models.'+args.gen_model)
    gen_net = eval('models.'+args.gen_model+'.CSformer')(args).cuda()
    print(f'model: {args.gen_model}')
    print(f'model param {(sum(param.numel() for param in  gen_net.parameters()))/1e6}M')
    print(f'dataset: {args.dataset}')
    print(f'cs ratio: {args.cs_ratio}')
    print(f'windows size: {args.g_window_size}')
    print(f'transformer depth: {[int(i) for i in args.g_depth.split(",")]}')
    print(f'transformer num_heads: {[int(i) for i in args.num_heads.split(",")]}')
    print(f'dim: {args.gf_dim}')
    print(f'CNN Norm: {args.cnnnorm_type}')
    print(f'Training patch size: {args.train_patch_size}')

    # weight init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    gen_net.apply(weights_init)
    gpu_ids = [i for i in range(int(torch.cuda.device_count()))]
    print('gpu ids:',gpu_ids)
    gen_net = torch.nn.DataParallel(gen_net.to("cuda:0"), device_ids=gpu_ids)


    # set optimizer
    if args.lr_multi is not None:
        print('multi lr is not None')
        gs_params_id = list(map(id, gen_net.module.gs.parameters()))
        decoder_params = filter(lambda p: id(p) not in gs_params_id, gen_net.parameters())
        gen_optimizer = torch.optim.Adam([{'params':gen_net.module.gs.parameters(), 'lr': args.g_lr},
                                          {'params': decoder_params, 'lr': args.decoder_lr}], betas=(args.beta1, args.beta2))
    else:
        gen_optimizer = torch.optim.Adam(gen_net.parameters(),
                                        args.g_lr, betas=(args.beta1, args.beta2))

    # set up data_loader
    dataset = datasets.ImageDataset(args)
    train_loader = dataset.train

    # epoch number
    args.max_epoch = args.max_epoch
    if args.max_iter:
        args.max_epoch = np.ceil(args.max_iter / len(train_loader))


    # ------------------------------  Cosine  LR ------------------------------
    t_max = args.max_epoch
    print(f'max epoch is {t_max}')
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(gen_optimizer, T_max=t_max, eta_min=1e-6)

    # initial
    start_epoch = 0
    best_psnr =10
    best_ssim =0
    psnr_score = 8
    best_epoch = 0

    # set up loss
    loss_all = get_loss_dict(args)
    print('loss:',loss_all.keys())
    print(f'rec_loss type: {args.rec_loss_type}\ng_lr：{args.g_lr}')
    # set writer
    if args.load_path:
        print(f'=> resuming from {args.load_path}')
        assert os.path.exists(args.load_path)
        checkpoint_file = os.path.join(args.load_path, 'Model', 'checkpoint_best.pth')
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch']
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        logger.info(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
    else:
        # create new log dir
        assert args.exp_name
        args.path_helper = set_log_dir('logs', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])

    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }

    # train loop
    for epoch in tqdm(range(int(start_epoch), int(args.max_epoch)), desc='total progress'):
        #lr
        cur_lr = gen_optimizer.param_groups[0]['lr']
        print(f'epoch[{epoch}] lr: {cur_lr}')
        writer = writer_dict['writer']
        writer.add_scalar('LR/g_lr', cur_lr, epoch)
        #lr
        start_time = time.time()
        train(args, gen_net,gen_optimizer,train_loader, epoch, writer_dict,
              loss_all)
        print(f'training time is {time.time()-start_time}')

        if epoch % args.val_freq == 0 or epoch == int(args.max_epoch)-1:

            backup_param = copy_params(gen_net)
            psnr_score,PSNR_cross, ssim_score,SSIM_cross = validate(args, epoch, gen_net, writer_dict)
            logger.info(f'@ epoch {epoch} || PSNR score: {psnr_score:.2f} SSIM score: {ssim_score:.4f}.\t @ best epoch {best_epoch} || Best PSNR score: {best_psnr:.2f} SSIM score: {best_ssim:.4f} .')
            load_params(gen_net, backup_param)

            if psnr_score > best_psnr:
                best_epoch = epoch
                best_psnr = psnr_score
                is_best = True
                best_ssim = ssim_score
                logger.info(f'@ epoch {epoch} || Best PSNR score: {psnr_score:.2f} SSIM score: {ssim_score:.4f}.')
            else:
                is_best = False
        else:
            is_best = False

        save_checkpoint({
            'epoch': epoch + 1,
            'gen_state_dict': gen_net.state_dict(),
            'best_psnr': best_psnr,
            'best_ssim': best_ssim,
            'path_helper': args.path_helper
        }, is_best, args.path_helper['ckpt_path'])

        #lr
        scheduler_lr.step()



if __name__ == '__main__':
    main()