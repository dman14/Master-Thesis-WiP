import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import PIL.Image as pil_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

def imshow(image, ax=None, title=None, normalize=False, size = (5,5)):
  """Imshow for Tensor."""
  if ax is None:
    fig, ax = plt.subplots(figsize=size)
  try:
    image = image.numpy().transpose((1, 2, 0))
  except:
    trans = transforms.ToTensor()
    image = trans(image)
    image = image.numpy().transpose((1, 2, 0))

  if normalize:
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

  ax.imshow(image)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.tick_params(axis='both', length=0)
  ax.set_xticklabels('')
  ax.set_yticklabels('')

  return ax

def save_img(img, name="saved_img", path=None, form=".png"):
  if path:
    dest = path + name + form
  else:
    dest = name + form

  try:
    img.save(dest)
  except AttributeError:
    trans = transforms.ToPILImage()
    trans(img).save(dest)


class Rescaler(object):
  """
  Rescaler class for rescaling images by a wanted factor.
  reupscale= flag for upscaling the downscaled images back to the original size.
  single = flag to return only on image, 'lr' or 'hr' instead of both
  """
  def __init__(self, scale = 4, reupscale= None, single = None):
    self.scale = scale
    self.reupscale = reupscale
    self.single = single

  def __call__(self, image):
    to_pil_image = transforms.ToPILImage()

    try:
      hr = to_pil_image(image)
    except:
      hr = image
    hr_width = (hr.width // self.scale) * self.scale
    hr_height = (hr.height // self.scale) * self.scale

    # Resizing hr image by rounding the width and height to be divisible
    if (hr_width != hr.width) or (hr_height != hr.height):
      hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)

    lr = hr.resize((hr_width // self.scale, hr_height // self.scale),
                    resample=pil_image.BICUBIC)
    if self.reupscale:
      lr = lr.resize((lr.width * self.scale, lr.height * self.scale),
                      resample=pil_image.BICUBIC)

    pil_to_tensor = transforms.ToTensor()(hr).unsqueeze_(0)
    tensor_to_pil = transforms.ToPILImage()(pil_to_tensor.squeeze_(0))
    hr = pil_to_tensor

    pil_to_tensor2 = transforms.ToTensor()(lr).unsqueeze_(0)
    tensor_to_pil2 = transforms.ToPILImage()(pil_to_tensor2.squeeze_(0))
    lr = pil_to_tensor2

    if self.single == "lr":
      return lr
    elif self.single == "hr":
      return hr
    else:
      return (lr,hr)