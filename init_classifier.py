
import matplotlib.pyplot as plt
from data.dataset import *

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import *
from pytorch_lightning.loggers import WandbLogger

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import wandb
import datetime
import os
from init_config import CLSConfig
import PIL
from torchsummary import summary

    

class Classifier(pl.LightningModule):
    
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()

        self.input_size = (lambda size: size[0] * size[1] * size[2])(input_dim)
        
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.input_size, 1024),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(1024, num_classes))
    """
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4*4*512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
    """    
    def forward(self, x):
        B, C, W, H = x.size()
        size = C * W * H
        x = x.view(-1, size)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    """
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    """
    def training_step(self, batch, batch_idx):
        imgs = batch['img']
        labels = batch['labels']
        pred = self(imgs)
        gt = torch.where(labels > 0,
                        torch.ones_like(labels).float(),
                        torch.zeros_like(labels).float())
        loss = nn.functional.binary_cross_entropy_with_logits(pred, gt)
        self.log('loss/training', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        imgs = batch['img']
        labels = batch['labels']
        pred = self(imgs)
        gt = torch.where(labels > 0,
                        torch.ones_like(labels).float(),
                        torch.zeros_like(labels).float())
        loss = nn.functional.binary_cross_entropy_with_logits(pred, gt)
        self.log('loss/validation', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        imgs = batch['img']
        indexs = batch['index']
        labels = batch['labels']
        pred = self(imgs)
        gt = torch.where(labels > 0,
                        torch.ones_like(labels).float(),
                        torch.zeros_like(labels).float())
        loss = nn.functional.binary_cross_entropy_with_logits(pred, gt)

        # Calculate accuracy
        preds_binary = torch.sigmoid(pred) > 0.5
        correct = torch.sum(preds_binary == gt)
        accuracy = correct.item() / torch.numel(gt)

        # Get the wrong classified images per class
        wrong_indices = torch.nonzero(preds_binary != gt, as_tuple=True)
        wrong_indices_per_class = wrong_indices[0][wrong_indices[1] == CelebHQAttrDataset.cls_to_id['Smiling']] #Look up in the id_to_cls from dataset
        gt_from_misclassified = [gt[wrong_indices_per_class][i][CelebHQAttrDataset.cls_to_id['Smiling']].item() for i in range(len(wrong_indices_per_class))]
        wrong_global_img_indexs = indexs[wrong_indices_per_class]

        self.log('loss/testing', loss)
        self.log('accuracy/testing', accuracy)

        for image, global_img_index, gt_misclassified in zip(imgs[wrong_indices_per_class], wrong_global_img_indexs, gt_from_misclassified):
            # Convert tensor to numpy array and then to PIL Image
            if global_img_index.item() < 28000:
                image_tensor = (image + 1) / 2
                image_array = image_tensor.permute(1, 2, 0).cpu().numpy()
                image_pil = Image.fromarray((image_array * 255).astype(np.uint8))

                # Define the image filename
                image_filename = os.path.join("/home/dai/GPU-Student-2/Cederic/DataSciPro/data/misclsData", f"{global_img_index}_{gt_misclassified}_misclassified.png")

                # Save the image as a PNG file
                image_pil.save(image_filename)

        images = [wandb.Image(image, caption=f"Idx: {str(global_img_index)}, gt: {str(gt_misclassified)}") for image, global_img_index, gt_misclassified in zip(imgs[wrong_indices_per_class], wrong_global_img_indexs, gt_from_misclassified)]
        wandb.log({"misclassified images": images})

        #table = wandb.Table(data=wrong_global_img_indexs, columns=["Wrong Image Indices"])
        #self.log({"Wrong Image Indices": table})

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=config.lr, weight_decay=config.weight_decay)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def trainCLS(
        config,
        data,
        num_cls,
): 
    train_size = int(0.8 * len(data))
    val_size = int(0.1 * len(data))
    test_size = len(data) - train_size - val_size

    train_dataset = Subset(data, range(train_size))
    val_dataset = Subset(data, range(train_size, train_size + val_size))
    test_dataset = Subset(data, range(train_size + val_size, train_size + val_size + test_size))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    print("Train set size:", len(train_dataset))
    print("Validation set size:", len(val_dataset))
    print("Test set size:", len(test_dataset))
    
    model = Classifier(data[0]['img'].shape, num_cls)
    #model = Classifier(num_cls)
    #summary(model, 256)
    wandb.log({'total_parameters': count_parameters(model)})
    
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    ckpt_callback = ModelCheckpoint(
        dirpath="/home/dai/GPU-Student-2/Cederic/pjds_group8/cls_checkpoints",
        filename="ffhq256." + formatted_time,
        monitor="loss/validation",
        mode="min",
        save_top_k=1,
        save_last=True,
        save_weights_only=True,
    )
    trainer = pl.Trainer(
        callbacks=[ckpt_callback],
        accelerator="gpu",
        devices=config.devices,
        max_epochs=config.epochs,
        #enable_model_summary=True,
        #enable_checkpointing=True,
        logger=WandbLogger(),
        log_every_n_steps=1,
        #strategy="ddp",
        num_nodes=1,
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    
    return trainer.checkpoint_callback.best_model_path
    
def main(config, data, num_cls):
    
    wandb.init(name=config.run_name, project="DataSciPro", entity="cedimac00", config=config)
    best_model_path = trainCLS(config, data, num_cls)
    print(best_model_path)
    wandb.finish()
    

if __name__ == "__main__":
    
    config = CLSConfig()
    data_path = '/home/dai/GPU-Student-2/Cederic/pjds_group8/datasets/celebahq256.lmdb'
    attr_path = '/home/dai/GPU-Student-2/Cederic/pjds_group8/datasets/celeba_anno/CelebAMask-HQ-attribute-anno.txt'
    image_size = 256
    original_resolution = 256
    num_cls = len(CelebHQAttrDataset.id_to_cls)

    data = CelebHQAttrDataset(path=data_path, image_size=image_size, attr_path=attr_path, do_augment=True)
    #fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    #ax[0].imshow(data[29996]['img'].permute(1, 2, 0).cpu())
    #ax[1].imshow(data[29996]['label'].permute(1, 2, 0).cpu())
    #plt.show()
    
    main(config, data, num_cls)
