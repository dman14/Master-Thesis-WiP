import os
import numpy as np
import torch
import math
from torch import nn, Tensor
from torch.nn.functional import softplus
from torch.distributions import Distribution

from torch import nn, Tensor
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from scripts.img_helper import *

class SRDataset(Dataset):
  """Super Resolution dataset."""

  def __init__(self, root_dir, transform=None, rescaler=None):
    """
    Args:
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
        on a sample.
   """
    self.images_frame = os.listdir(root_dir)
    self.root_dir = root_dir
    self.transform = transform
    if rescaler:
      self.rescaler = rescaler
    else:
      self.rescaler = Rescaler()

  def __len__(self):
    return len(self.images_frame)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()


    img_name = os.path.join(self.root_dir,
                            self.images_frame[idx])
    image = Image.open(img_name)

    if self.transform:
      image = self.transform(image)

    aux = self.rescaler(image)
    if self.rescaler.single:
      sample = aux
    else:
      sample = {'lr': aux[0], 'hr': aux[1]}

    return sample

def SRTransform(size):
  transform = transforms.Compose([transforms.Resize(size),
                                 transforms.CenterCrop(size),
                                 transforms.ToTensor()])
  return transform

class SRDataLoader(DataLoader):
  def __init__(self, path,
              scale=4,reupscale =None,
              single= None, size=64,
              batch_size=4, shuffle=True,
               num_workers=0):
    
    self.transform = self.init_transform(size)
    self.rescaler = self.init_rescaler(scale, reupscale, single)
    self.dataset = self.init_dataset(path, self.transform, self.rescaler)
    self.dataloader = self.init_dataloader(self.dataset,batch_size,
                                           shuffle,num_workers)

  def init_transform(self,size):
    return SRTransform(size)

  def init_rescaler(self, scale, reupscale, single):
    return Rescaler(scale, reupscale, single)

  def init_dataset(self, path, transform, rescaler):
    return SRDataset(path, transform, rescaler)

  def init_dataloader(self,dataset,batch_size,shuffle,num_workers):
    return DataLoader(dataset, batch_size=2,
                      shuffle=True, num_workers=0)
  def get_dataloader(self):
    return self.dataloader
