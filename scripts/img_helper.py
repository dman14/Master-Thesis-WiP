from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import PIL.Image as pil_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from math import ceil, floor
import pandas as pd
from pathlib import Path
import cv2
from skimage.metrics import structural_similarity as ssim
import torch.nn as nn


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
    #trans = transforms.ToPILImage()
    #trans(img).save(dest)
    cv2.imwrite(dest, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


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
      
    hr = hr.convert(mode='RGB')
    
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

def deriveSizeFromScale(img_shape, scale):
    output_shape = []
    for k in range(2):
        output_shape.append(int(ceil(scale[k] * img_shape[k])))
    return output_shape
 
def deriveScaleFromSize(img_shape_in, img_shape_out):
    scale = []
    for k in range(2):
        scale.append(1.0 * img_shape_out[k] / img_shape_in[k])
    return scale
 
def triangle(x):
    x = np.array(x).astype(np.float64)
    lessthanzero = np.logical_and((x>=-1),x<0)
    greaterthanzero = np.logical_and((x<=1),x>=0)
    f = np.multiply((x+1),lessthanzero) + np.multiply((1-x),greaterthanzero)
    return f
 
def cubic(x):
    x = np.array(x).astype(np.float64)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    f = np.multiply(1.5*absx3 - 2.5*absx2 + 1, absx <= 1) + np.multiply(-0.5*absx3 + 2.5*absx2 - 4*absx + 2, (1 < absx) & (absx <= 2))
    return f
 
def contributions(in_length, out_length, scale, kernel, k_width):
    if scale < 1:
        h = lambda x: scale * kernel(scale * x)
        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x = np.arange(1, out_length+1).astype(np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = int(ceil(kernel_width)) + 2
    ind = np.expand_dims(left, axis=1) + np.arange(P) - 1 # -1 because indexing from 0
    indices = ind.astype(np.int32)
    weights = h(np.expand_dims(u, axis=1) - indices - 1) # -1 because indexing from 0
    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]
    return weights, indices
 
def imresizemex(inimg, weights, indices, dim):
    in_shape = inimg.shape
    w_shape = weights.shape
    out_shape = list(in_shape)
    out_shape[dim] = w_shape[0]
    outimg = np.zeros(out_shape)
    if dim == 0:
        for i_img in range(in_shape[1]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[ind, i_img].astype(np.float64)
                outimg[i_w, i_img] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)
    elif dim == 1:
        for i_img in range(in_shape[0]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[i_img, ind].astype(np.float64)
                outimg[i_img, i_w] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)        
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg

def imresizevec(inimg, weights, indices, dim):
    wshape = weights.shape
    if dim == 0:
        weights = weights.reshape((wshape[0], wshape[2], 1, 1))
        outimg =  np.sum(weights*((inimg[indices].squeeze(axis=1)).astype(np.float64)), axis=1)
    elif dim == 1:
        weights = weights.reshape((1, wshape[0], wshape[2], 1))
        outimg =  np.sum(weights*((inimg[:, indices].squeeze(axis=2)).astype(np.float64)), axis=2)
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg
 
def resizeAlongDim(A, dim, weights, indices, mode="vec"):
    if mode == "org":
        out = imresizemex(A, weights, indices, dim)
    else:
        out = imresizevec(A, weights, indices, dim)
    return out
 
def imresize(I, scalar_scale=None, method='bicubic', output_shape=None, mode="vec"):
    if method is 'bicubic':
        kernel = cubic
    elif method is 'bilinear':
        kernel = triangle
    else:
        print ('Error: Unidentified method supplied')
        
    kernel_width = 4.0
    # Fill scale and output_size
    if scalar_scale is not None:
        scalar_scale = float(scalar_scale)
        scale = [scalar_scale, scalar_scale]
        output_size = deriveSizeFromScale(I.shape, scale)
    elif output_shape is not None:
        scale = deriveScaleFromSize(I.shape, output_shape)
        output_size = list(output_shape)
    else:
        print ('Error: scalar_scale OR output_shape should be defined!')
        return
    scale_np = np.array(scale)
    order = np.argsort(scale_np)
    weights = []
    indices = []
    for k in range(2):
        w, ind = contributions(I.shape[k], output_size[k], scale[k], kernel, kernel_width)
        weights.append(w)
        indices.append(ind)
    B = np.copy(I) 
    flag2D = False
    if B.ndim == 2:
        B = np.expand_dims(B, axis=2)
        flag2D = True
    for k in range(2):
        dim = order[k]
        B = resizeAlongDim(B, dim, weights[dim], indices[dim], mode)
    if flag2D:
        B = np.squeeze(B, axis=2)
    return B
 
def convertDouble2Byte(I):
    B = np.clip(I, 0.0, 1.0)
    B = 255*B
    return np.around(B).astype(np.uint8)

def rgb2ycbcr(im_rgb):
 im_rgb = im_rgb.astype(np.float32)
 im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
 im_ycbcr = im_ycrcb[:,:,(0,2,1)].astype(np.float32)
 im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*(235-16)+16)/255.0 #to [16/255, 235/255]
 im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*(240-16)+16)/255.0 #to [16/255, 240/255]
 return im_ycbcr

def quality_measure_YCbCr(target, output):

  #target = (target.numpy().transpose((1, 2, 0))*255).astype(np.uint8)
  #output = (output.numpy().transpose((1, 2, 0))*255).astype(np.uint8)
  #target = (target.transpose((1, 2, 0))*255).astype(np.uint8)
  #output = (output.transpose((1, 2, 0))*255).astype(np.uint8)
  #target = np.asarray(target)
  #output = np.asarray(output)

  target_ycbcr = rgb2ycbcr(target)
  output_ycbcr = rgb2ycbcr(output)
 
  y1 = target_ycbcr[:,:,0]
  y2 = output_ycbcr[:,:,0]
 
  psnr = cv2.PSNR(y1, y2) 
  #score, diff = ssim(y1.astype(np.uint8), y2.astype(np.uint8), multichannel= False, full = True)
  score = ssim_basicsr(y1, y2)

  return psnr, score


def ssim_basicsr(img, img2):
  """Calculate SSIM (structural similarity) for one channel images.
  It is called by func:`calculate_ssim`.
  Args:
      img (ndarray): Images with range [0, 255] with order 'HWC'.
      img2 (ndarray): Images with range [0, 255] with order 'HWC'.
  Returns:
      float: ssim result.
    """

  c1 = (0.01 * 255)**2
  c2 = (0.03 * 255)**2

  img = img.astype(np.float64)
  img2 = img2.astype(np.float64)
  kernel = cv2.getGaussianKernel(11, 1.5)
  window = np.outer(kernel, kernel.transpose())

  mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
  mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
  mu1_sq = mu1**2
  mu2_sq = mu2**2
  mu1_mu2 = mu1 * mu2
  sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
  sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
  sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

  ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
  return ssim_map.mean()

#Making Patches
def make_patches(tensor,patch_size=16,scale=4):
  #mask = torch.ones_like(tensor)
  tensor = tensor.unsqueeze(0)
  stride  = patch_size//2
  wo = tensor.size(2)
  ho = tensor.size(3)
  wn = wo + stride - (wo%stride)
  hn = ho + stride - (ho%stride)
  if stride - (wo%stride) > 0 and stride - (wo%stride) < stride:
    #pad = nn.ReplicationPad2d((0,0,0,wn-wo))
    #tensor = pad(tensor)
    tensor = np.pad(tensor,((0,0),(0,0),(0,wn-wo),(0,0)),mode='edge')
    tensor = torch.from_numpy(tensor)
  else:
    wn = wo
  if stride - (ho%stride) > 0 and stride - (ho%stride) < stride:
    #pad = nn.ReplicationPad2d((0,hn-ho,0,0))
    #tensor = pad(tensor)
    tensor = np.pad(tensor,((0,0),(0,0),(0,0),(0,hn-ho)),mode='edge')
    tensor = torch.from_numpy(tensor)
  else:
    hn = ho
  mask= torch.ones((tensor.size()[0],tensor.size()[1],tensor.size()[2]*scale,tensor.size()[3]*scale))
  # use torch.nn.Unfold
  unfold  = nn.Unfold(kernel_size=(patch_size, patch_size), stride=stride)
  unfold2 = nn.Unfold(kernel_size=(patch_size*scale, patch_size*scale), stride=stride*scale)
  # Apply to mask and original image
  mask_p  = unfold2(mask)
  patches = unfold(tensor)

  patches = patches.reshape(3, patch_size, patch_size, -1).permute(3, 0, 1, 2)
  if tensor.is_cuda:
      patches_base = torch.zeros((patches.size()[0],patches.size()[1],patches.size()[2]*scale,patches.size()[3]*scale), device=tensor.get_device())
  else: 
      patches_base = torch.zeros((patches.size()[0],patches.size()[1],patches.size()[2]*scale,patches.size()[3]*scale))

  tiles = []
  for t in range(patches.size(0)):
       tiles.append(torch.squeeze(patches[[t], :, :, :]))
  return tiles, mask_p, patches_base, (tensor.size(2)*scale, tensor.size(3)*scale), ((wn-wo)*scale,(hn-ho)*scale)

#Putting Patches back together
def image_from_patches(tensor_list, mask_t, base_tensor, t_size, tile_size=256, padding=0):
    stride  = tile_size//2
    base_tensor = base_tensor.to('cuda')
    # base_tensor here is used as a container

    for t, tile in enumerate(tensor_list):
         #print(tile.size())
         #tile = torch.from_numpy(tile)
         tile = tile.type(torch.FloatTensor)
         tile = tile.to('cuda')
         base_tensor[[t], :, :] = tile
   
    base_tensor = base_tensor.permute(1, 2, 3, 0).reshape(3*tile_size*tile_size, base_tensor.size(0)).unsqueeze(0)
    fold = nn.Fold(output_size=(t_size[0], t_size[1]), kernel_size=(tile_size, tile_size), stride=stride)
    # https://discuss.pytorch.org/t/seemlessly-blending-tensors-together/65235/2?u=bowenroom
    mask_t = mask_t.to('cuda')
    output_tensor = fold(base_tensor)/fold(mask_t)

    output_tensor = output_tensor[:,:, :output_tensor.shape[2]-padding[0], :output_tensor.shape[3]-padding[1]]
    return output_tensor