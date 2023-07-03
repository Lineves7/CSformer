# CSformer: Bridging Convolution and Transformer for Compressive Sensing (TIP 2023)

Official Pytorch implementation of "**CSformer: Bridging Convolution and Transformer for Compressive Sensing**" published in ***IEEE Transactions on Image Processing (TIP)***.
#### [[Paper-arXiv](https://arxiv.org/abs/2112.15299)] [[Paper-official](https://ieeexplore.ieee.org/document/10124835/)] 
Dongjie Ye, [Zhangkai Ni](https://eezkni.github.io/), [Hanli Wang](https://mic.tongji.edu.cn/51/91/c9778a86417/page.htm), [Jian Zhang](https://jianzhang.tech/), [Shiqi Wang](https://www.cs.cityu.edu.hk/~shiqwang/), [Sam Kwong](http://www6.cityu.edu.hk/stfprofile/cssamk.htm)



## Testing (Running pretrained models)
- Checkpoint

Checkpoints can be found from [Google Drive](https://drive.google.com/file/d/1P_HKhmTsYi2H94VMY1TcIU5Ze6H_mIq0/view?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1o7Cs9OLjy63PLydgFmQ_qw?pwd=fr6m) (提取码：fr6m). 

- Inference
1. Unzip the checkpoint file and place all the files in the ./logs/checkpoint_coco/ directory.
2. Edit the ./cfg.py file to modify the [--testdata_path] by specifying the path to your test datasets.
3. Excute the test script below:
    ```
    python eval.py --cs_ratio 1 --exp_name test_CS1 --load_path ./logs/checkpoint_coco/checkpoint_CS1.pth --overlap --overlapstep 8
    ```
   (The available options for [cs_ratio] in our pre-trained model are 1, 4, 10, 25, and 50.)
      
    If you want to test the model wihtout overlapping, you may run the script below:
    ```
    python eval.py --cs_ratio 1 --exp_name test_CS1 --load_path ./logs/checkpoint_coco/checkpoint_CS1.pth
    ```
## Training (Coming Soon)

## Citation
If this code is useful for your research, please cite our paper:

```
@article{csformer,
  author={Ye, Dongjie and Ni, Zhangkai and Wang, Hanli and Zhang, Jian and Wang, Shiqi and Kwong, Sam},
  journal={IEEE Transactions on Image Processing}, 
  title={CSformer: Bridging Convolution and Transformer for Compressive Sensing}, 
  year={2023},
  volume={32},
  number={},
  pages={2827-2842},
  doi={10.1109/TIP.2023.3274988}}
```

## Contact

Thanks for your attention! If you have any suggestion or question, feel free to leave a message here or contact Dongjie Ye (dj.ye@my.cityu.edu.hk).
