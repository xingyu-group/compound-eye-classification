import torch
import torchvision
import torch.utils.data as data
from PIL import Image
import glob
import numpy as np
import os

def pil_loader(path):
    return Image.open(path).convert('RGB')

class CompoundEyeDataset(data.Dataset):
    def __init__(self, data_root=None, loader=pil_loader, num_img_per_folder=7400, train=True, transforms=None):
        self.train = train
        self.seed = 14
        self.data_root = data_root
        self.loader = loader
        self.num_img_per_folder = num_img_per_folder
        self.fog_strength_split = 10
        self.transforms = transforms
        self.reset_random_state()
        self.random_split_data(ratio=0.9)

    def reset_random_state(self):
        self.rand_state = np.random.RandomState(self.seed)
    
    def random_split_data(self, ratio=0.9):
        image_list = []
        labels_list = []
        label = 0
        for i in ['A','E','I','O','U']:
            temp_image_list = []
            if i in ['A','E','U']:
                folder_num = self.rand_state.choice(8, 4)
            else:
                folder_num = [0, 1, 2, 3]
            for j in folder_num:
                tmp = glob.glob(os.path.join(self.data_root, 'Compound-eye-classification', i, i+str(j+1), '*.png'))
                if len(tmp) < self.num_img_per_folder:
                    temp_image_list += tmp
                temp_image_list += tmp[:self.num_img_per_folder]
            labels_list += [label]*len(temp_image_list)
            image_list += temp_image_list
            label += 1
        idx = self.rand_state.permutation(len(image_list))
        image_list = np.array(image_list)
        labels_list = np.array(labels_list)
        train_list = image_list[idx[:int(len(idx)*ratio)]]
        train_labels = labels_list[idx[:int(len(idx)*ratio)]]
        test_list = image_list[idx[int(len(idx)*ratio):]]
        test_labels = labels_list[idx[int(len(idx)*ratio):]]
        
        if self.train:
            self.imgs = train_list
            self.labels = train_labels
        else:
            self.test_data_fog_split = [[] for _ in range(self.fog_strength_split)]
            self.test_label_fog_split = [[] for _ in range(self.fog_strength_split)]
            for i in range(len(test_list)):
                img_num = int(test_list[i][-10:-4])-1
                fog_strength = img_num//(self.num_img_per_folder//self.fog_strength_split)
                if fog_strength == self.fog_strength_split:
                    fog_strength -= 1
                self.test_data_fog_split[fog_strength].append(test_list[i])
                self.test_label_fog_split[fog_strength].append(test_labels[i])
            self.imgs = test_list
            self.labels = test_labels
    
    def set_test_fog_strength(self, fog_strength):
        assert self.train == False
        assert fog_strength <= self.fog_strength_split and fog_strength > 0
        self.imgs = self.test_data_fog_split[fog_strength-1]
        self.labels = self.test_label_fog_split[fog_strength-1]
        self.imgs = np.array(self.imgs)
        self.labels = np.array(self.labels)

    def reset_test_fog_strength(self):
        assert self.train == False
        self.imgs = []
        self.labels = []
        for i in range(len(self.test_data_fog_split)):
            self.imgs += self.test_data_fog_split[i]
            self.labels += self.test_label_fog_split[i]
        self.imgs = np.array(self.imgs)
        self.labels = np.array(self.labels)
        
    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.transforms(self.loader(path))
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.imgs)
