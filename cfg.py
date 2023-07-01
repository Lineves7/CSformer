import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # train
    parser.add_argument(
        '--max_epoch',
        type=int,
        default=200,
        help='number of epochs of training')
    parser.add_argument(
        '--max_iter',
        type=int,
        default=None,
        help='set the max iteration number')
    parser.add_argument(
        '--g_lr',
        type=float,
        default=0.0001,
        help='adam: gen learning rate')
    parser.add_argument(
        '--lr_decay',
        action='store_true',
        help='learning rate decay or not')
    parser.add_argument(
        '--beta1',
        type=float,
        default=0.0,
        help='adam: decay of first order momentum of gradient')
    parser.add_argument(
        '--beta2',
        type=float,
        default=0.9,
        help='adam: decay of first order momentum of gradient')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='number of cpu threads to use during batch generation')
    parser.add_argument(
        '--val_freq',
        type=int,
        default=20,
        help='interval between each validation')

    parser.add_argument(
        '--dataset',
        type=str,
        default='coco',
        help='dataset type')
    parser.add_argument(
        '--data_path',
        type=str,
        default="D:\database\package\coco\\unlabeled2017",
        help='The path of data set')
    parser.add_argument('--init_type', type=str, default='normal',
                        choices=['normal', 'orth', 'xavier_uniform', 'false'],
                        help='The init type')
    parser.add_argument('--optimizer', type=str, default="adam",
                        help='optimizer')
    parser.add_argument('--rec_w', type=int, default=1, help='penalty for the reconstruction loss')
    parser.add_argument('--rec_loss_type', type=str, default='l1', help='The type of reconstruction loss')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='penalty for the adam')
    parser.add_argument('--lr_multi', type=int, default=None)
    parser.add_argument('--train_patch_size',type=int,default=128,help='size of training input in dataloader')
    parser.add_argument('--random_seed', type=int, default=12345)

    # setting
    parser.add_argument(
        '--exp_name',
        type=str,
        help='The name of exp')
    parser.add_argument(
        '--load_path',
        type=str,
        help='The reload model path')
    parser.add_argument('--cs_ratio', type=int, default=10, help='from {1, 4, 10, 25, 40, 50}')
    parser.add_argument('--datarange', type=str, default='-11',
                        help='input data norm to range')
    parser.add_argument('--torch_vision', action='store_true', default=False, help='Show intermediate results in tensorbard dir')

    #test
    parser.add_argument(
        '--testdata_path',
        type=str,
        default=[r'C:\dataset\data\Set11'],
        # default=['D:\database\dataset\\Urban100','C:\dataset\data\Set11','D:\database\dataset\BSD68','D:\database\dataset\Set14','D:\database\dataset\Set5\data'],
        help='The path of data set')
    parser.add_argument('--eval_batch_size', type=int, default=400)
    parser.add_argument('--overlap',  action='store_true',help='overlap or not during testing')
    parser.add_argument('--overlapstep',type=int,default=8,help='the overlap step for testing')

    #model
    parser.add_argument(
        '--gen_model',
        type=str,
        default='CSformer',
        help='path of gen model')
    parser.add_argument(
        '--bottom_width',
        type=int,
        default=8)
    parser.add_argument(
        '--img_size',
        type=int,
        default=64,
        help='output size')
    parser.add_argument('--gf_dim', type=int, default=128,
                        help='The base channel num of gen')

    parser.add_argument('--g_depth', type=str, default="5,5,5,5",
                        help='Generator Depth')
    parser.add_argument('--g_window_size', type=int, default=8,
                        help='generator mlp ratio')
    parser.add_argument('--num_heads', type=str, default="16,8,4,2",
                        help='num_head of transformer')
    parser.add_argument('--cnnnorm_type', type=str, default="BatchNorm",
                        help='norm type of cnn')
    parser.add_argument('--g_norm', type=str, default="ln",
                        help='Generator Normalization')
    parser.add_argument('--g_mlp', type=int, default=4,
                        help='generator mlp ratio')
    parser.add_argument('--g_act', type=str, default="gelu",
                        help='Generator activation Layer')
    parser.add_argument('--pretrained', type=str, default="church",
                        help='pretrained dataset')
    parser.add_argument('--seed', default=12345, type=int,
                        help='seed for initializing training. ')


    opt = parser.parse_args()

    return opt
