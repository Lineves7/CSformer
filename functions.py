
import torch.nn as nn
from imageio import imsave
from tqdm import tqdm
from torchvision import utils as vutils
import torchvision.transforms as transforms
import glob
from utils.utils import *
from torch.autograd import Variable
from torch.utils.data import BatchSampler,SequentialSampler
from torch.utils.data._utils.collate import default_collate as collate_fn
import time
from copy import deepcopy

logger = logging.getLogger(__name__)

def train(args, gen_net: nn.Module, gen_optimizer,train_loader,epoch, writer_dict, loss_all):
    writer = writer_dict['writer']
    gen_step = 0
    # train mode
    gen_net = gen_net.train()

    for iter_idx, imgs in enumerate(tqdm(train_loader)):
        if imgs.size(0) != args.gen_batch_size:
            continue

        global_steps = writer_dict['train_global_steps']

        input_imgs = imgs.type(torch.cuda.FloatTensor).to("cuda:0")
        real_imgs = input_imgs

        # ---------------------
        #  Training
        # ---------------------
        gen_optimizer.zero_grad()
        gen_imgs,init_rgb,tf_patch,ini_patch = gen_net(input_imgs)
        #
        if iter_idx%200==0:
            if args.datarange == '-11':
                final_rgb2 = torch.squeeze(gen_imgs[1, :, :, :])
                init_rgb2 = torch.squeeze(init_rgb[1,:,:,:])
                gt_rgb2 = torch.squeeze(real_imgs[1,:,:,:])
                final_rgb2 = np.round((final_rgb2.detach().cpu().numpy() + 1.) * 127.5)
                init_rgb2 = np.round((init_rgb2.detach().cpu().numpy() + 1.) * 127.5)
                gt_rgb2 = np.round((gt_rgb2.detach().cpu().numpy() + 1.) * 127.5)
            # torchvision
            if args.torch_vision == True:
                checkrgb = torch.from_numpy(np.stack((gt_rgb2, final_rgb2, init_rgb2), axis=0))
                checkrgb = checkrgb.unsqueeze(1)
                img_grid = vutils.make_grid(checkrgb, nrow=3, normalize=True, range=(0, 255))
                writer.add_image("train_img___gt___output___ini", img_grid, global_steps)

        #  cal loss
        rec_loss = args.rec_w * loss_all['rec_loss'](gen_imgs, real_imgs)
        loss = rec_loss
        rec_loss_print = rec_loss.item()
        writer.add_scalar('rec_loss', rec_loss_print, global_steps)

        loss.backward()
        gen_optimizer.step()

        gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            tqdm.write(
                    "[Epoch %d/%d] [Batch %d/%d] [loss: %f] [Rec loss: %f]" %
                    (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), loss.item(),
                     rec_loss_print))

        writer_dict['train_global_steps'] = global_steps + 1

def validate(args, epoch, gen_net: nn.Module, writer_dict):
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    # eval mode
    gen_net = gen_net.eval()
    PSNR_cross = []
    SSIM_cross = []


    with torch.no_grad():
        val_set_path = args.valdata_path
        val_set_path1 = glob.glob(val_set_path + '/*.tif')
        val_set_path2 = glob.glob(val_set_path + '/*.png')
        val_set_path = val_set_path1 + val_set_path2
        ImgNum = len(val_set_path)
        PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
        SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)

        dname = 'val'
        stime = time.time()


        for img_no in tqdm(range(ImgNum)):
            imgName = val_set_path[img_no]
            Iorg = imread_CS_py_new(imgName)
            if args.datarange == '-11':
                Ifloat = Iorg/ 127.5 - 1.
            inputs = Variable(torch.from_numpy(Ifloat.astype('float32')).cuda())
            inputs = torch.unsqueeze(inputs, dim=0)
            inputs = torch.unsqueeze(inputs, dim=0)
            _, _, h_old, w_old = inputs.size()
            padding_block = 64
            if h_old%padding_block != 0:
                h_pad = (h_old // padding_block+1) * padding_block - h_old
            else:
                h_pad = 0
            if w_old%padding_block != 0:
                w_pad = (w_old // padding_block+1) * padding_block - w_old
            else:
                w_pad = 0
            inputs = torch.cat([inputs, torch.flip(inputs, [2])], 2)[:, :, :h_old + h_pad, :]
            inputs = torch.cat([inputs, torch.flip(inputs, [3])], 3)[:, :, :, :w_old + w_pad]


            output,ini_rgbs, _, _ = gen_net(inputs)

            output = torch.squeeze(output)
            Inirec = torch.squeeze(ini_rgbs)


            output = output.cpu().data.numpy()
            Inirec = Inirec.cpu().data.numpy()


            images_recovered = output[0:h_old, 0:w_old]
            Inirec = Inirec[0:h_old, 0:w_old]

            if args.datarange == '-11':
                images_recovered = np.clip(images_recovered, -1, 1).astype(np.float32)
                Irec = np.round((images_recovered+1.) * 127.5)


            rec_PSNR = psnr(Irec, Iorg)
            PSNR_All[0, img_no] = rec_PSNR
            rec_SSIM = compute_ssim(Irec, Iorg)
            SSIM_All[0, img_no] = rec_SSIM

        #
        PSNR_mean = np.mean(PSNR_All)
        PSNR_cross.append(PSNR_mean)
        SSIM_mean = np.mean(SSIM_All)
        SSIM_cross.append(SSIM_mean)
        tqdm.write(
            "[%s] [Epoch %d] [Mean PSNR: %f] [Mean SSIM: %f] [time: %d]" %
            (dname, epoch, PSNR_mean, SSIM_mean,time.time()-stime))

    PSNR_cross_mean = np.mean(PSNR_cross)
    SSIM_cross_mean = np.mean(SSIM_cross)

    writer.add_scalar('val_SSIM_score', SSIM_cross_mean, global_steps)
    writer.add_scalar('val_PSNR_score', PSNR_cross_mean, global_steps)

    writer_dict['valid_global_steps'] = global_steps + 1

    return PSNR_cross_mean,PSNR_cross,SSIM_cross_mean,SSIM_cross

def test(args,gen_net: nn.Module, logger):
    # eval mode
    gen_net = gen_net.eval()

    PSNR_cross = []
    SSIM_cross = []
    stage1_PSNR_cross = []
    stage1_SSIM_cross = []

    with torch.no_grad():
        for i in range(len(args.testdata_path)):
        # for i in range(1):
            test_set_path = args.testdata_path[i]
            test_set_path1 = glob.glob(test_set_path + '/*.tif')
            test_set_path2 = glob.glob(test_set_path + '/*.png')
            test_set_path = test_set_path1 + test_set_path2
            ImgNum = len(test_set_path)
            PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
            SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
            PSNR_stage1ALL = np.zeros([1, ImgNum], dtype=np.float32)
            SSIM_stage1ALL = np.zeros([1, ImgNum], dtype=np.float32)

            if ImgNum == 11:
                dname = 'set11'
            elif ImgNum == 68:
                dname = 'BSD68'
            elif ImgNum == 14:
                dname = 'set14'
            elif ImgNum == 5:
                dname = 'set5'
            elif ImgNum == 100:
                dname = 'urban100'
            else:
                dname = 'test'



            save_dir = args.path_helper['sample_path']
            print(f'save dir is {save_dir}')

            if not os.path.exists(os.path.join(save_dir)):
                os.makedirs(os.path.join(save_dir))

            stime = time.time()
            for img_no in tqdm(range(ImgNum)):
                imgName = test_set_path[img_no]
                [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(imgName)
                if args.datarange == '-11':
                    # Icol = img2col_py(Ipad, 64) / 127.5 - 1.  # uint to [-1,1]
                    Ipad = Ipad / 127.5 - 1.
                inputs = Variable(torch.from_numpy(Ipad.astype('float32')).cuda())
                inputs = torch.unsqueeze(inputs, dim=0)
                inputs = torch.unsqueeze(inputs, dim=0)

                output, _, _,_ = gen_net(inputs)

                output = torch.squeeze(output)


                output = output.cpu().data.numpy()


                images_recovered = output[0:row, 0:col]

                if args.datarange == '-11':
                    Irec = np.round((images_recovered + 1.) * 127.5)

                rec_PSNR = psnr(Irec, Iorg)
                PSNR_All[0, img_no] = rec_PSNR


                rec_SSIM = compute_ssim(Irec, Iorg)
                SSIM_All[0, img_no] = rec_SSIM


                imgname_for_save = os.path.basename(imgName)
                imgname_for_save = os.path.splitext(imgname_for_save)[0]
                imgname_for_save = imgname_for_save + '.png'
                imgname_for_save = os.path.join(save_dir,imgname_for_save)
                imsave(imgname_for_save,Irec.astype(np.uint8))

        PSNR_mean = np.mean(PSNR_All)
        PSNR_cross.append(PSNR_mean)
        SSIM_mean = np.mean(SSIM_All)
        SSIM_cross.append(SSIM_mean)


        logger.info(f"[{dname}] [Mean PSNR loss: {PSNR_mean:.2f}]  [Mean SSIM loss: {SSIM_mean:.4f}] )")


    PSNR_cross_mean = np.mean(PSNR_cross)
    SSIM_cross_mean = np.mean(SSIM_cross)


    logger.info(f"[all cross] [Mean PSNR loss: {PSNR_cross_mean}]  [Mean SSIM loss: {SSIM_cross_mean}] ")


def test_overlap(args,gen_net: nn.Module, logger):
    # eval mode
    gen_net = gen_net.eval()
    step = args.overlapstep
    PSNR_cross = []
    SSIM_cross = []
    logger.info(f'the overlap step is {step}')
    with torch.no_grad():
        #for i in range(len(args.testdata_path)):
        for i in range(1):
            test_set_path = args.testdata_path[i]
            # test_set_path = args.testdata_path
            print(f'test_set_path is {test_set_path} \n')

            test_set_path1 = glob.glob(test_set_path + '/*.tif')
            test_set_path2 = glob.glob(test_set_path + '/*.png')
            test_set_path3 = glob.glob(test_set_path + '/*.JPG')
            test_set_path4 = glob.glob(test_set_path + '/*.jpg')
            test_set_path = test_set_path1 + test_set_path2 + test_set_path3 + test_set_path4
            ImgNum = len(test_set_path)
            PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
            SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
            # PSNR_stage1ALL = np.zeros([1, ImgNum], dtype=np.float32)
            # SSIM_stage1ALL = np.zeros([1, ImgNum], dtype=np.float32)
            print(f'len is {ImgNum} \n')

            if ImgNum == 11:
                dname = 'set11'
            elif ImgNum == 68:
                dname = 'BSD68'
            elif ImgNum == 14:
                dname = 'set14'
            elif ImgNum == 5:
                dname = 'set5'
            elif ImgNum == 100:
                dname = 'urban100'

            save_dir = args.path_helper['sample_path']
            print(f'save dir is {save_dir}')



            for img_no in tqdm(range(ImgNum)):
                imgName = test_set_path[img_no]
                [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(imgName)
                if args.datarange == '-11':
                    patches = img2patches(Ipad, (64, 64), (step,step))
                    patches_batch = patches / 127.5 - 1.
                inputs = Variable(torch.from_numpy(patches_batch.astype('float32')).cuda())
                inputs = inputs.permute(0,3,1,2)


                output = torch.FloatTensor(inputs.shape[0],inputs.shape[1],inputs.shape[2],inputs.shape[3]).cuda()
                ini_rgbs = torch.FloatTensor(inputs.shape[0],inputs.shape[1],inputs.shape[2],inputs.shape[3]).cuda()

                batch_list = list(BatchSampler(SequentialSampler(output), batch_size=args.eval_batch_size, drop_last=False))
                for idx, list_data in enumerate(BatchSampler(inputs, batch_size=args.eval_batch_size, drop_last=False)):
                    batch_x = collate_fn(list_data)
                    list_tmp = batch_list[idx]
                    output[list_tmp, :, :, :], _, _, _ = gen_net(batch_x)



                output = output.permute(0,2,3,1)
                Inirec = ini_rgbs.permute(0,2,3,1)


                output = output.cpu().data.numpy()
                Inirec = Inirec.cpu().data.numpy()

                #unpatch
                output = unpatch2d(output, Ipad.shape,(step,step)).squeeze()
                Inirec = unpatch2d(Inirec, Ipad.shape,(step,step)).squeeze()

                images_recovered = output[0:row, 0:col]
                Inirec = Inirec[0:row, 0:col]

                if args.datarange == '-11':
                    Irec = np.round((images_recovered + 1.) * 127.5)


                rec_PSNR = psnr(Irec, Iorg)
                PSNR_All[0, img_no] = rec_PSNR

                rec_SSIM = compute_ssim(Irec, Iorg)
                SSIM_All[0, img_no] = rec_SSIM

                if not os.path.exists(os.path.join(save_dir)):
                    os.makedirs(os.path.join(save_dir))

                imgname_for_save = os.path.basename(imgName)
                imgname_for_save = os.path.splitext(imgname_for_save)[0]
                imgname_for_save = imgname_for_save + '.png'
                imgname_for_save = os.path.join(save_dir,imgname_for_save)
                imsave(imgname_for_save,Irec.astype(np.uint8))


            PSNR_mean = np.mean(PSNR_All)
            PSNR_cross.append(PSNR_mean)
            SSIM_mean = np.mean(SSIM_All)
            SSIM_cross.append(SSIM_mean)


            logger.info(f"[{dname}] [Mean PSNR loss: {PSNR_mean:.2f}]  [Mean SSIM loss: {SSIM_mean:.4f}] )")


    PSNR_cross_mean = np.mean(PSNR_cross)
    SSIM_cross_mean = np.mean(SSIM_cross)


    logger.info(f"[all cross] [Mean PSNR loss: {PSNR_cross_mean}]  [Mean SSIM loss: {SSIM_cross_mean}] ")



def pytorch_unnormalize(tensor,mean=(0.5,0.5,0.5),std = (0.5,0.5,0.5)):
    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )
    return inv_normalize(tensor)

def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten