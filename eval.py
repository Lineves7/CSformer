import cfg
import torch
from functions import test_overlap,test
from utils.utils import set_log_dir, create_logger
import os
from tensorboardX import SummaryWriter


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)
    assert args.exp_name
    assert args.load_path.endswith('.pth')
    assert os.path.exists(args.load_path)
    args.path_helper = set_log_dir('logs_eval', args.exp_name)
    logger = create_logger(args.path_helper['log_path'], phase='test')


    # Get Sensing_matrix
    # ratio_dict = {1: 10, 4: 43, 10: 103, 25: 272, 30: 327, 40: 436, 50: 545} #32*32
    ratio_dict = {1: 3, 4: 11, 10: 26, 25: 64, 30: 77, 40: 103, 50: 128} #16*16
    n_input = ratio_dict[args.cs_ratio]
    args.n_input = n_input

    # import network
    exec('import '+'models.'+args.gen_model)
    gen_net = eval('models.'+args.gen_model+'.CSformer')(args).cuda()
    gpu_ids = [i for i in range(int(torch.cuda.device_count()))]
    print('gpu ids:',gpu_ids)
    gen_net = torch.nn.DataParallel(gen_net.to("cuda:0"), device_ids=gpu_ids)
    print(f'gen_model: {args.gen_model}')
    print(f'Model params: {sum(param.numel() for param in  gen_net.parameters())}')
    print(f'dataset: {args.dataset}')
    print(f'cs ratio: {args.cs_ratio}')


    # set writer
    logger.info(f'=> resuming from {args.load_path}')
    print(f'=> resuming from {args.load_path}')
    checkpoint_file = args.load_path
    assert os.path.exists(checkpoint_file)
    checkpoint = torch.load(checkpoint_file)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
    }
    writer = writer_dict['writer']
    if 'avg_gen_state_dict' in checkpoint:
        gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        epoch = checkpoint['epoch']
        logger.info(f'=> loaded checkpoint {checkpoint_file} (epoch {epoch})')
    else:
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        logger.info(f'=> loaded checkpoint {checkpoint_file}')
        print(f'=> loaded checkpoint {checkpoint_file}')

    if args.overlap == True:
        test_overlap(args,gen_net,logger)
    else:
        test(args,gen_net,logger)


if __name__ == '__main__':
    main()