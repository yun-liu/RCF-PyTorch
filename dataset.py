import torch
import cv2
import numpy as np
import os.path as osp


class BSDS_Dataset(torch.utils.data.Dataset):
    def __init__(self, root='data/HED-BSDS', split='train', transform=False):
        self.root = root
        self.split = split
        self.transform = transform
        if self.split == 'train':
            self.file_list = osp.join(self.root, 'train_pair.lst')
        elif self.split == 'test':
            self.file_list = osp.join(self.root, 'test.lst')
        else:
            raise ValueError('Invalid split type!')
        with open(self.file_list, 'r') as f:
            self.file_list = f.readlines()
        self.mean = np.array([0.406, 0.456, 0.485], dtype=np.float32)
        self.std  = np.array([0.225, 0.224, 0.229], dtype=np.float32)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if self.split == 'train':
            img_file, label_file = self.file_list[index].split()
            label = cv2.imread(osp.join(self.root, label_file), 0)
            label = np.array(label, dtype=np.float32)
            label = label[np.newaxis, :, :]
            label[label == 0] = 0
            label[np.logical_and(label > 0, label < 127.5)] = 2
            label[label >= 127.5] = 1
        else:
            img_file = self.file_list[index].rstrip()

        img = cv2.imread(osp.join(self.root, img_file))
        img = np.array(img, dtype=np.float32)
        img = (img / 255 - self.mean) / self.std
        img = img[:, :, ::-1].copy().transpose((2, 0, 1))

        if self.split == 'train':
            return img, label
        else:
            return img
