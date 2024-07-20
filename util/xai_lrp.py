import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_attr(attr, norm='abs', clamp_neg=False, aggregate_channels='sum', cmap='seismic', text=None, color_level=0.9, show=False):
      # expects a tensor attr of shape (C x H x W), ie one value per voxel.

      # to numpy
      attr = attr.squeeze(0).detach().numpy()

      if clamp_neg:
            attr = np.maximum(attr, 0)

      if aggregate_channels: # sum, abs_sum, l2
            # assumes the existance of channel axis at tensor axis index 0
            if aggregate_channels == 'sum':
                  attr = attr.sum(axis=0)
            elif aggregate_channels == 'abs_sum':
                  attr = np.abs(attr).sum(axis=0)
            elif aggregate_channels == 'l2':
                  attr = np.sqrt((attr*attr).sum(0))

      if norm == 'abs':
            attr /= np.abs(attr).max() # attr to range [-1,1]
            attr = (attr + 1)/2        # attr to range [0,1]


      # colorize attributions. assumes input values within [0,1].
      cmap = matplotlib.colormaps[cmap]
      attr_shape = attr.shape
      attr_img = cmap(np.reshape(attr, [np.prod(attr.shape)])) # returns RGBA
      attr_img = np.reshape(attr_img[...,0:3], attr_shape + (3,))# reshape RGB information (only) to original attribution shape
      attr_img = (attr_img * color_level * 255).astype(np.uint8)

      if show:
          plt.imshow(attr_img, cmap=cmap, vmin=0, vmax=1)
          if text: 
               plt.title(text)
          #plt.show()


      return attr_img

def show_attributions(path, attrs, title=None, **kwargs):
    # display the attributions. the **kwargs are passed to the visualization helper viszalize_attr
    fig, axes = plt.subplots(1, len(attrs), figsize=(15, 3), squeeze=False)
    for i in range(len(attrs)):
            ax = axes[0,i]
            attr_img = visualize_attr(attrs[i].cpu(), **kwargs)
            ax.imshow(attr_img)
            ax.axis('off')
            ax.set_title(title)

    fig.savefig(f'{path}/xai_{title}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    #plt.show()
    return attr_img

def xai_zennit(model, x, RuleComposite, device, target=None):
  # the model does not require special implementations to be Explainable
  with RuleComposite.context(model): # this places backward hooks (temporarily) replacing / modifying the vanilla gradient backward pass
    x.requires_grad = True
    y_pred = model(x)
    #if not target: target = y

    # initiate relevance
    relevance_init = y_pred *  torch.eye(np.prod(y_pred.shape))[target].to(device) # this again populates
    relevance, = torch.autograd.grad(y_pred, x, relevance_init) # this uses the autograd functionality to execute LRP

  return relevance, target


