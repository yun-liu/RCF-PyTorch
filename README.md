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
    
### Training and Testing

1. Clone the RCF repository
    ```
    git clone https://github.com/yun-liu/RCF-PyTorch.git
    ```

2. Download [the ImageNet-pretrained model](https://drive.google.com/file/d/1szqDNG3dUO6BM3l6YBuC9vWp16n48-cK/view?usp=sharing), and put it into the `$ROOT_DIR` folder.

3. Download the datasets you need as below, and extract these datasets to the `$ROOT_DIR/data/` folder.

    ```
    wget http://mftp.mmcheng.net/liuyun/rcf/data/bsds_pascal_train_pair.lst
    wget http://mftp.mmcheng.net/liuyun/rcf/data/HED-BSDS.tar.gz
    wget http://mftp.mmcheng.net/liuyun/rcf/data/PASCAL.tar.gz
    ```
    
4. Modify the path to the dataset and the path to the output folder in `train.sh`, and then, run the following command for running the code:
    ```
    ./train.sh
    ```
    
For more information about RCF, please refer to this page: [yun-liu/RCF](https://github.com/yun-liu/RCF)

### We have released the code and data for plotting the edge PR curves of many existing edge detectors [here](https://github.com/yun-liu/plot-edge-pr-curves).

