import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import PIL.Image as pil_image

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


def rescale(image, scale = 4, reupscale= None, single = None):
  to_pil_image = transforms.ToPILImage()

  try:
    hr = to_pil_image(image)
  except:
    hr = image
  hr_width = (hr.width // scale) * scale
  hr_height = (hr.height // scale) * scale

  # Resizing hr image by rounding the width and height to be divisible
  hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)

  lr = hr.resize((hr_width // scale, hr_height // scale), resample=pil_image.BICUBIC)
  if reupscale:
    lr = lr.resize((lr.width * scale, lr.height * scale), resample=pil_image.BICUBIC)

  pil_to_tensor = transforms.ToTensor()(hr).unsqueeze_(0)
  tensor_to_pil = transforms.ToPILImage()(pil_to_tensor.squeeze_(0))
  hr = pil_to_tensor

  pil_to_tensor2 = transforms.ToTensor()(lr).unsqueeze_(0)
  tensor_to_pil2 = transforms.ToPILImage()(pil_to_tensor2.squeeze_(0))
  lr = pil_to_tensor2

  if single == "lr":
    return lr
  elif single == "hr":
    return hr
  else:
    return (lr,hr)