import os
import numpy as np
import torch
import math
from torch import nn, Tensor
from torch.nn.functional import softplus
from torch.distributions import Distribution

from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class SRDataset(Dataset):
  """Super Resolution dataset."""

  def __init__(self, root_dir, transform=None):
    """
    Args:
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
        on a sample.
   """
    self.images_frame = os.listdir(root_dir)
    self.root_dir = root_dir
    self.transform = transform
    self.set_rescaler()

  def __len__(self):
    return len(self.images_frame)

  def set_rescaler(self, scale=4, reupscale =None, single= None):
    self.scale = scale
    self.reupscale = None
    self.single = None

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()


    img_name = os.path.join(self.root_dir,
                            self.images_frame[idx])
    image = Image.open(img_name)

    if self.transform:
      image = self.transform(image)

    aux = rescale(image, self.scale, self.reupscale, self.single)
    if self.single:
      sample = aux
    else:
      sample = {'lr': aux[0], 'hr': aux[1]}

    return sample

def SRTransform(size):
  transform = transforms.Compose([transforms.Resize(size),
                                 transforms.CenterCrop(size),
                                 transforms.ToTensor()])
  return transform