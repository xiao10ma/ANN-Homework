import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import ipdb

class faceDataset(Dataset):
    def __init__(self, **kwargs):
        super(faceDataset, self).__init__()
        data_root, split = kwargs['data_root'], kwargs['split']
        image_size = kwargs['image_size']
        self.data_root = os.path.join(data_root, split)
        
        self.faces = []

        print('Loading Angry Faces')
        angry_path = os.path.join(self.data_root, 'Angry')
        for filename in tqdm(os.listdir(angry_path)):
            img = Image.open(os.path.join(angry_path, filename))
            img  = img.resize(image_size, Image.Resampling.LANCZOS)
            img = np.asarray(img)
            img = img.astype(np.float32) / 255.0
            self.faces.append((img.transpose(2, 0, 1), torch.tensor([1.0, 0, 0, 0, 0])))
        
        print('Loading Happy Faces')
        happy_path = os.path.join(self.data_root, 'Happy')
        for filename in tqdm(os.listdir(happy_path)):
            img = Image.open(os.path.join(happy_path, filename))
            img  = img.resize(image_size, Image.Resampling.LANCZOS)
            img = np.asarray(img)
            img = img.astype(np.float32) / 255.0
            self.faces.append((img.transpose(2, 0, 1), torch.tensor([0, 1.0, 0, 0, 0])))
        
        print('Loading Neutral Faces')
        neutral_path = os.path.join(self.data_root, 'Neutral')
        for filename in tqdm(os.listdir(neutral_path)):
            img = Image.open(os.path.join(neutral_path, filename))
            img  = img.resize(image_size, Image.Resampling.LANCZOS)
            img = np.asarray(img)
            img = img.astype(np.float32) / 255.0
            self.faces.append((img.transpose(2, 0, 1), torch.tensor([0, 0, 1.0, 0, 0])))

        print('Loading Sad Faces')
        sad_path = os.path.join(self.data_root, 'Sad')
        for filename in tqdm(os.listdir(sad_path)):
            img = Image.open(os.path.join(sad_path, filename))
            img  = img.resize(image_size, Image.Resampling.LANCZOS)
            img = np.asarray(img)
            img = img.astype(np.float32) / 255.0
            self.faces.append((img.transpose(2, 0, 1), torch.tensor([0, 0, 0, 1.0, 0])))

        print('Loading Surprise Faces')
        surprise_path = os.path.join(self.data_root, 'Surprise')
        for filename in tqdm(os.listdir(surprise_path)):
            img = Image.open(os.path.join(surprise_path, filename))
            img  = img.resize(image_size, Image.Resampling.LANCZOS)
            img = np.asarray(img)
            img = img.astype(np.float32) / 255.0
            self.faces.append((img.transpose(2, 0, 1), torch.tensor([0, 0, 0, 0, 1.0])))

    def __getitem__(self, index):
        return self.faces[index]
    
    def __len__(self, ):
        return len(self.faces)