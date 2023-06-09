import time

import torch.utils.data as data
import numpy as np
import os
import pickle
from PIL import Image
# from skimage.transform import resize
from cv2 import imread
# from scipy.misc import imread, imresize
import torch

#判断路径是否存在
def check_exists(root):
    if os.path.exists(root):
        return True
    return False



class CUHKPEDES_BERT_token(data.Dataset):
    '''
    Args:
        root (string): Base root directory of dataset where [split].pkl and [split].h5 exists
        split (string): 'train', 'val' or 'test'
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed vector. E.g, ''transform.RandomCrop'
        target_transform (callable, optional): A funciton/transform that tkes in the
            targt and transfomrs it.
    '''

    #BERT
    pklname_list = ['/content/TIPCB/BERT_id_train_64_revised.npz', '/content/TIPCB/BERT_id_val_64_revised.npz',
                    '/content/TIPCB/BERT_id_test_64_revised.npz']



    def __init__(self, root, split, max_length, transform=None, target_transform=None,
                 cap_transform=None):

        self.root = root
        self.max_length = max_length
        self.transform = transform
        self.target_transform = target_transform
        self.cap_transform = cap_transform
        self.split = split.lower()  # 返回将字符串中所有大写字符转换为小写后生成的字符串。

        # if not check_exists(self.root):
        #     print(self.root)
        #     raise RuntimeError('Dataset not found or corrupted.' +
        #                        'Please follow the directions to generate datasets')

        if self.split == 'train':
            self.pklname = self.pklname_list[0]

            with open(os.path.join("/content/TIPCB/", self.pklname), 'rb') as f_pkl:
                data = pickle.load(f_pkl)
                self.train_labels = [int(i)-1 for i in data['labels']]
                self.train_captions = data['caption_id']
                self.train_images = data['images_path']
                self.train_attention_mask = data['attention_mask']
            # data_h5py = h5py.File(os.path.join(self.root, self.h5name), 'r')
            # self.train_images = data_h5py['images']


        elif self.split == 'val':
            self.pklname = self.pklname_list[1]
            with open(os.path.join("/content/TIPCB/", self.pklname), 'rb') as f_pkl:
                data = pickle.load(f_pkl)
                self.val_labels = [int(i) - 11004 for i in data['labels']]
                self.val_captions = data['caption_id']
                self.val_images = data['images_path']
                self.val_attention_mask = data['attention_mask']
            # data_h5py = h5py.File(os.path.join(self.root, self.h5name), 'r')
            # self.val_images = data_h5py['images']

        elif self.split == 'test':
            self.pklname = self.pklname_list[2]

            with open(os.path.join("/content/TIPCB/", self.pklname), 'rb') as f_pkl:
                data = pickle.load(f_pkl)
                self.test_labels = [int(i) -12004 for i in data['labels']]
                self.test_captions = data['caption_id']
                self.test_images = data['images_path']
                self.test_attention_mask = data['attention_mask']

            # data_h5py = h5py.File(os.path.join(self.root, self.h5name), 'r')
            # self.test_images = data_h5py['images']

        else:
            raise RuntimeError('Wrong split which should be one of "train","val" or "test"')

    def __getitem__(self, index):
        """
        Args:
              index(int): Index
        Returns:
              tuple: (images, labels, captions)
        """

        if self.split == 'train':
            img_path, caption, attention_mask, label = 'CUHK-PEDES/imgs/'+self.train_images[index], self.train_captions[index], \
                                                      self.train_attention_mask[index],self.train_labels[index]
        elif self.split == 'val':
            img_path, caption, attention_mask, label = 'CUHK-PEDES/imgs/'+self.val_images[index], self.val_captions[index],\
                                                      self.val_attention_mask[index],self.val_labels[index]
        else:
            img_path, caption, attention_mask, label = 'CUHK-PEDES/imgs/'+self.test_images[index], self.test_captions[index],\
                                                       self.test_attention_mask[index], self.test_labels[index]
        img_path = os.path.join(self.root, img_path)
        img = imread(img_path)

        if len(img.shape) == 2:
            img = np.dstack((img, img, img))
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)
        label=torch.tensor(label)

        if self.cap_transform is not None:
            caption = self.cap_transform(caption)

        # caption = caption[1:-1]
        caption = np.array(caption)
        attention_mask = np.array(attention_mask)
        if len(caption) >= self.max_length:
            caption = caption[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        else:
            pad = np.zeros((self.max_length - len(caption), 1), dtype=np.int64)
            caption = np.append(caption, pad)
            attention_mask = np.append(attention_mask, pad)
        caption = torch.tensor(caption).long()
        attention_mask = torch.tensor(attention_mask).long()
        return img, caption, label, attention_mask


    def __len__(self):
        if self.split == 'train':
            return len(self.train_labels)
        elif self.split == 'val':
            return len(self.val_labels)
        else:
            return len(self.test_labels)
