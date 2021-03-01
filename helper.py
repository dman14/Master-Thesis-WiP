import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

def imshow(image, ax=None, title=None, normalize=False, size = (10,10)):
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

    #ax.tight_layout()
    return ax