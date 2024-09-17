## [Richer Convolutional Features for Edge Detection](http://mmcheng.net/rcfedge/)

This is the PyTorch implementation of our edge detection method, RCF.

### Citations

If you are using the code/model/data provided here in a publication, please consider citing:

    @article{liu2019richer,
      title={Richer Convolutional Features for Edge Detection},
      author={Liu, Yun and Cheng, Ming-Ming and Hu, Xiaowei and Bian, Jia-Wang and Zhang, Le and Bai, Xiang and Tang, Jinhui},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
      volume={41},
      number={8},
      pages={1939--1946},
      year={2019},
      publisher={IEEE}
    }

    @article{liu2022semantic,
      title={Semantic edge detection with diverse deep supervision},
      author={Liu, Yun and Cheng, Ming-Ming and Fan, Deng-Ping and Zhang, Le and Bian, JiaWang and Tao, Dacheng},
      journal={International Journal of Computer Vision},
      volume={130},
      pages={179--198},
      year={2022},
      publisher={Springer}
    }
    
### Training

1. Clone the RCF repository:
    ```
    git clone https://github.com/yun-liu/RCF-PyTorch.git
    ```

2. Download the ImageNet-pretrained model ([Google Drive](https://drive.google.com/file/d/1szqDNG3dUO6BM3l6YBuC9vWp16n48-cK/view?usp=sharing) or [Baidu Yun](https://pan.baidu.com/s/1vfntX-cTKnk58atNW5T1lA?pwd=g5af)), and put it into the `$ROOT_DIR` folder.

3. Download the datasets as below, and extract these datasets to the `$ROOT_DIR/data/` folder.

    ```
    wget http://mftp.mmcheng.net/liuyun/rcf/data/bsds_pascal_train_pair.lst
    wget http://mftp.mmcheng.net/liuyun/rcf/data/HED-BSDS.tar.gz
    wget http://mftp.mmcheng.net/liuyun/rcf/data/PASCAL.tar.gz
    ```
    
4. Run the following command to start the training:
    ```
    python train.py --save-dir /path/to/output/directory/
    ```
    
### Testing

1. Download the pretrained model (BSDS500+PASCAL: [Google Drive](https://drive.google.com/file/d/1oxlHQCM4mm5zhHzmE7yho_oToU5Ucckk/view?usp=sharing) or [Baidu Yun](https://pan.baidu.com/s/1Tpf_-dIxHmKwH5IeClt0Ng?pwd=03ad)), and put it into the `$ROOT_DIR` folder.

2. Run the following command to start the testing:
    ```
    python test.py --checkpoint bsds500_pascal_model.pth --save-dir /path/to/output/directory/
    ```
   This pretrained model should achieve an ODS F-measure of 0.812.

For more information about RCF and edge quality evaluation, please refer to this page: [yun-liu/RCF](https://github.com/yun-liu/RCF)

### Edge PR Curves

We have released the code and data for plotting the edge PR curves of many existing edge detectors [here](https://github.com/yun-liu/plot-edge-pr-curves).

### RCF based on other frameworks 

Caffe based RCF: [yun-liu/RCF](https://github.com/yun-liu/RCF)

Jittor based RCF: [yun-liu/RCF-Jittor](https://github.com/yun-liu/RCF-Jittor)

### Acknowledgements

[1] [balajiselvaraj1601/RCF_Pytorch_Updated](https://github.com/balajiselvaraj1601/RCF_Pytorch_Updated)

[2] [meteorshowers/RCF-pytorch](https://github.com/meteorshowers/RCF-pytorch)
