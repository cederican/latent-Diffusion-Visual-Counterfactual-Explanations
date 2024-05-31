from dataset import CelebHQAttrDataset
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

data_path = '/home/dai/GPU-Student-2/Cederic/pjds_group8/datasets/celebahq256.lmdb'
attr_path = '/home/dai/GPU-Student-2/Cederic/pjds_group8/datasets/celeba_anno/CelebAMask-HQ-attribute-anno.txt'
image_size = 256
data = CelebHQAttrDataset(path=data_path, image_size=image_size, attr_path=attr_path, do_augment=False)

index = 27046
image_tensor = (data[index]['img'] + 1) / 2

# Convert the tensor to a numpy array and then to a PIL Image
image_array = image_tensor.permute(1, 2, 0).cpu().numpy()
image_pil = Image.fromarray((image_array * 255).astype(np.uint8))

# Save the image as a PNG file
image_pil.save(f"/home/dai/GPU-Student-2/Cederic/DataSciPro/data/misclsData_gt1/{index}_misclassified.png")

