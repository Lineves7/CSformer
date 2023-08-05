C:/Anaconda3/envs/pytorch_17/python train.py \
--gen_model CSformer \
--exp_name coco_cs4\
--cs_ratio 4 \
--img_size 64 \
--bottom_width 8 \
--max_iter 500000 \
--g_lr 1e-4 \
--gen_batch_size 10 \
--eval_batch_size 10 \
--gf_dim 128 \
--val_freq 1 \
--print_freq 100 \
--g_window_size 8 \
--num_workers 1 \
--optimizer adam  \
--beta1 0.9 \
--beta2 0.999 \
--init_type xavier_uniform \
--g_depth 5,5,5,5 \
--datarange -11 \
--train_patch_size 128 \
--rec_loss_type l2 \
--dataset coco \
--data_path D:\database\package\coco\\unlabeled2017
# --dataset BSD400 \
# --data_path C:/dataset/data/BSD400


