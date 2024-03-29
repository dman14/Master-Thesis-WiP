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
    #self.images_frame = os.listdir(root_dir)
    self.images_frame = [os.path.join(pth, f)
                        for pth, dirs, files in os.walk(root_dir) for f in files]
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


    #img_name = os.path.join(self.root_dir,
    #                        self.images_frame[idx])
    img_name = self.images_frame[idx]
    image = Image.open(img_name)

    if self.transform:
      image = self.transform(image)

    aux = self.rescaler(image)
    if self.rescaler.single:
      sample = aux
    else:
      #sample = {'lr': aux[0], 'hr': aux[1]}
      sample = (aux[0],aux[1])

    return sample

def SRTransform(size=64):
  """
  Function to return a wanted transform for a dataset
  needs size to which to resize the images
  """
  transform = transforms.Compose([transforms.Resize(size),
                                 transforms.CenterCrop(size),
                                 transforms.ToTensor()])
  return transform

class SRDataLoader(DataLoader):
  """
  Dataloader class to bring up and initialize a dataloader,
  transform, dataset and a rescaler
  """
  def __init__(self, path,
              scale=4,reupscale =None,
              single= None, size=64,
              batch_size=4, shuffle=True,
               num_workers=0):
    self.batch_size = batch_size
    self.shuffle = shuffle 
    self.num_workers = num_workers
    self.init_transform(size)
    self.init_rescaler(scale, reupscale, single)
    self.init_dataset(path, self.transform, self.rescaler)

  def init_transform(self,size):
    self.transform = SRTransform(size)

  def init_rescaler(self, scale, reupscale, single):
    self.rescaler = Rescaler(scale, reupscale, single)

  def init_dataset(self, path, transform, rescaler):
    self.dataset = SRDataset(path, transform, rescaler)

  def get_dataloader(self):
    return DataLoader(self.dataset, batch_size=self.batch_size,
                      shuffle=self.shuffle,
                      num_workers=self.num_workers)




class SRDataset_patches(Dataset):
  """Super Resolution dataset."""

  def __init__(self, root_dir, transform=None, rescaler=None, size = 16,repatch_data = False):
    """
    Args:
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
        on a sample.
   """
    #self.images_frame = os.listdir(root_dir)
    self.images_frame = [os.path.join(pth, f)
                        for pth, dirs, files in os.walk(root_dir) for f in files]
    self.root_dir = root_dir
    self.transform = transform
    self.size = size
    self.repatch_data = repatch_data
    if rescaler:
      self.rescaler = rescaler
    else:
      self.rescaler = Rescaler()

  def __len__(self):
    return len(self.images_frame)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()


    #img_name = os.path.join(self.root_dir,
    #                        self.images_frame[idx])
    img_name = self.images_frame[idx]
    image = Image.open(img_name)

    if self.transform:
      image = self.transform(image)

    aux = self.rescaler(image)
    if self.rescaler.single:
      sample, mask_p, patches_base, t_size, padding = make_patches(aux,self.size)
      
    else:
      #sample = {'lr': aux[0], 'hr': aux[1]}
      sample1, mask_p, patches_base, t_size, padding = make_patches(aux[0],self.size)
      sample2, mask_p2, patches_base2, t_size2, padding2 = make_patches(aux[1],self.size*4,scale=1)
      sample = (sample1,sample2)

    if self.repatch_data == False:
      return sample
    else:
      return sample, (mask_p, patches_base, t_size, padding), (mask_p2, patches_base2, t_size2, padding2)

def SRTransform_patches():
  """
  Function to return a wanted transform for a dataset
  needs size to which to resize the images
  """
  transform = transforms.Compose([transforms.ToTensor()])
  return transform

class SRDataLoader_patches(DataLoader):
  """
  Dataloader class to bring up and initialize a dataloader,
  transform, dataset and a rescaler
  """
  def __init__(self, path,
              scale=4,reupscale=None,
              single=None, batch_size=4, 
              size=16, shuffle=True,
              num_workers=0,repatch_data = False):
    self.batch_size = batch_size
    self.shuffle = shuffle 
    self.num_workers = num_workers
    self.size = size
    self.repatch_data = repatch_data
    self.init_transform_patches()
    self.init_rescaler(scale, reupscale, single)
    self.init_dataset_patches(path, self.transform, self.rescaler)

  def init_transform_patches(self):
    self.transform = SRTransform_patches()

  def init_rescaler(self, scale, reupscale, single):
    self.rescaler = Rescaler(scale, reupscale, single)

  def init_dataset_patches(self, path, transform, rescaler):
    self.dataset = SRDataset_patches(path, transform, rescaler, size = self.size, repatch_data = self.repatch_data)

  def get_dataloader_patches(self):
    return DataLoader(self.dataset, batch_size=self.batch_size,
                      shuffle=self.shuffle,
                      num_workers=self.num_workers,collate_fn=lambda x: x)
