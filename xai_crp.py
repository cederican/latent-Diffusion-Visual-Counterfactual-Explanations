from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from crp.image import imgify

from zennit.composites import EpsilonPlusFlat
from zennit.canonizers import SequentialMergeBatchNorm

import torch
import torch.nn as nn

from init_classifier import LinearClassifier, VQVAEClassifier, ResNet50Classifier
from data.dataset import ImageDataset, CelebHQAttrDataset
from torch.utils.data import DataLoader
from xai_lrp import xai_zennit, show_attributions
import matplotlib
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ResNet50Classifier.load_from_checkpoint("/home/dai/GPU-Student-2/Cederic/DataSciPro/cls_checkpoints/ffhq256.b64res502024-06-02 17:06:41.ckpt",
                                            num_classes = len(CelebHQAttrDataset.id_to_cls))
model.to(device)
model.eval()

data = ImageDataset('/home/dai/GPU-Student-2/Cederic/DataSciPro/data/misclsData_gt0', image_size=256, exts=['jpg', 'JPG', 'png'], do_augment=False, sort_names=True)
dataloader = DataLoader(data, batch_size=1, shuffle=False)
for batch in dataloader:
        image = batch['img'].to(model.device)

        # define LRP rules and canonizers in zennit
        composite = EpsilonPlusFlat([SequentialMergeBatchNorm()])

        # load CRP toolbox
        attribution = CondAttribution(model, no_param_grad=True)

        # here, each channel is defined as a concept
        # or define your own notion!
        cc = ChannelConcept()

        # get a conditional attribution for channel 50 in layer features.27 wrt. output 1
        conditions = [{ 'y': [31]}]

        image.requires_grad = True
        attr = attribution(image, conditions, composite, mask_map=cc.mask)

        layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])
        mask_map = {name: cc.mask for name in layer_names}

        attr = attribution(image, conditions, composite, mask_map=mask_map)

        from crp.image import imgify

        print(torch.equal(attr[0], attr.heatmap))

        imgify(attr.heatmap, symmetric=True)
