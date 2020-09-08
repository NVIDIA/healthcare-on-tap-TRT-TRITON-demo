"""
The MIT License (MIT)

Copyright (c) 2020 NVIDIA

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import json
import numpy as np
import os
from pathlib import Path
from PIL import Image


import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
from matplotlib import patches, patheffects
import matplotlib.gridspec as gridspec

class CXRDataset(Dataset):
    """CXR dataset."""
    def __init__(self, dataset_info, transform=None, is_training = True):
        """
        Args:
            dataset_info (string): Dataset JSON file with all the details
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.dataset_info = dataset_info['training']
        if not is_training:
            self.dataset_info = dataset_info['testing'][:8000]
        self.transform = transform

    def __len__(self):
        return len(self.dataset_info)

    def __getitem__(self, idx):
        img_name = self.dataset_info[idx]['image']
        image = np.array(Image.open(prescaled_path / img_name))
        if image.ndim == 3:
            image = image[:,:,0]
        image = torch.tensor(image).expand(3, 256, 256)
        label = np.asarray(self.dataset_info[idx]['label'], dtype='int8')
        if self.transform:
            image = self.transform(image)

        return (image,label)

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def get_test_loader_TTA(data, batch_size, workers=5, _worker_init_fn=None, shuffle=True, is_training=False):
    full_dataset = CXRDataset(data,transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.TenCrop((224)),
#             transforms.Resize((224,224)),
#             transforms.ToTensor() #Too slow
            #normalize,
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
        ]), is_training=is_training)
    
    test_loader = torch.utils.data.DataLoader(full_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True)
    return test_loader

def show_images(data,targs,labels,batch_size):
    columns = 4
    rows = (batch_size + 1) // (columns)
    fig = plt.figure(figsize = (16,(16 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows*columns):
        ax = plt.subplot(gs[j])
        plt.axis("off")
        rand_idx = np.random.randint((j-1)*(10),(j-1)*(10)+10)
        img = data[rand_idx].cpu().squeeze()
        targ_lab = [labels[str(idx)] for idx,xx in enumerate(targs[j]) if xx!=0]
        if targ_lab == []: targ_lab = ['Normal']
        ax = show_img_title(img[0], targ_lab, figsize=(10,10), ax=ax)


def show_img_title(im, targs, figsize=None, ax=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im, cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(targs, fontsize = 14)
    return ax

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))
            
prescaled_path = Path('/workspace/data/ChestXray14/images_prescaled/')
