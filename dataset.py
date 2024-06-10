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
        self.seed = 0
        self.data_root = data_root
        self.loader = loader
        self.num_img_per_folder = num_img_per_folder
        self.fog_strength_split = 10
        self.transforms = transforms
        self.reset_random_state()
        

    def reset_random_state(self):
        self.rand_state = np.random.RandomState(self.seed)
    
    def random_split_data(self):
        image_list = []
        for i in ['A','E','I','O','U']:
            if i in ['A','E','U']:
                folder_num = self.rand_state.choice(8, 4)
            else:
                folder_num = [0, 1, 2, 3]
            for j in folder_num:
                tmp = glob.glob(os.path.join(self.data_root, 'Compound-eye-classification', i, str(j+1), '*.png'))
                if len(tmp) < self.num_img_per_folder:
                    image_list += tmp
                image_list += tmp[:self.num_img_per_folder]
        self.rand_state.shuffle(image_list)
        train_list = image_list[:int(len(image_list)*0.9)]
        test_list = image_list[int(len(image_list)*0.9):]

        self.train_list = train_list
        self.test_list = test_list
        
        if self.train:
            self.imgs = train_list
        else:
            self.test_list_fog_split = [[] for _ in range(self.fog_strength_split)]
            for i in test_list:
                img_num = int(i[5:9])-1
                self.test_list_fog_split[img_num//(self.num_img_per_folder//self.fog_strength_split)].append(i)
            self.imgs = test_list
    
    def set_test_fog_strength(self, fog_strength):
        assert fog_strength <= self.fog_strength_split and fog_strength > 0
        self.imgs = self.test_list_fog_split[fog_strength-1]

    def reset_test_fog_strength(self):
        self.imgs = []
        for i in range(len(self.test_list_fog_split)):
            self.imgs += self.test_list_fog_split[i]

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.transforms(self.loader(path))
        return img

    def __len__(self):
        return len(self.imgs)
