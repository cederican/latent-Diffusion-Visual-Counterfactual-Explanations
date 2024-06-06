
import matplotlib.pyplot as plt
import PIL
import datetime
import os
import numpy as np
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.models as models

import pytorch_lightning as pl
from pytorch_lightning.callbacks import *
from pytorch_lightning.loggers import WandbLogger

from diffusers import VQModel

from data.dataset import *
from init_config import CLSConfig

class ResNet50Classifier(pl.LightningModule):
    def __init__(self, num_classes):
        super(ResNet50Classifier, self).__init__()

        # Initialize the ResNet50 model
        self.resnet50 = models.resnet50(pretrained=True)
        
        # Modify the last fully connected layer to match the number of classes
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.resnet50(x)

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
            if global_img_index.item() < 30000:
                image_tensor = (image + 1) / 2
                image_array = image_tensor.permute(1, 2, 0).cpu().numpy()
                image_pil = Image.fromarray((image_array * 255).astype(np.uint8))

                # Define the image filename
                image_filename = os.path.join("/home/dai/GPU-Student-2/Cederic/DataSciPro/data/misclsData_RESNET50", f"{global_img_index}_{gt_misclassified}_misclassified.png")

                # Save the image as a PNG file
                image_pil.save(image_filename)

        images = [wandb.Image(image, caption=f"Idx: {str(global_img_index)}, gt: {str(gt_misclassified)}") for image, global_img_index, gt_misclassified in zip(imgs[wrong_indices_per_class], wrong_global_img_indexs, gt_from_misclassified)]
        wandb.log({"misclassified images": images})
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=config.lr, weight_decay=config.weight_decay)  

class VQVAEClassifier(pl.LightningModule):
    
    def __init__(self, num_classes):
        super(VQVAEClassifier, self).__init__()

        self.vqvae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae")
    
        self.fc1 = nn.Linear(3*64*64, num_classes)
             
    def forward(self, x):
        z = self.vqvae.encode(x).latents
        B, C, W, H = z.size()
        size = C * W * H
        z = z.view(-1, size)
        z = self.fc1(z)
        return z
    
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
            if global_img_index.item() < 30000:
                image_tensor = (image + 1) / 2
                image_array = image_tensor.permute(1, 2, 0).cpu().numpy()
                image_pil = Image.fromarray((image_array * 255).astype(np.uint8))

                # Define the image filename
                image_filename = os.path.join("/home/dai/GPU-Student-2/Cederic/DataSciPro/data/misclsData_VQVAE", f"{global_img_index}_{gt_misclassified}_misclassified.png")

                # Save the image as a PNG file
                image_pil.save(image_filename)

        images = [wandb.Image(image, caption=f"Idx: {str(global_img_index)}, gt: {str(gt_misclassified)}") for image, global_img_index, gt_misclassified in zip(imgs[wrong_indices_per_class], wrong_global_img_indexs, gt_from_misclassified)]
        wandb.log({"misclassified images": images})
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=config.lr, weight_decay=config.weight_decay)   

class LinearClassifier(pl.LightningModule):
    
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()

        self.input_size = (lambda size: size[0] * size[1] * size[2])(input_dim)
        
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.input_size, 1024),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(1024, num_classes))
   
    def forward(self, x):
        B, C, W, H = x.size()
        size = C * W * H
        x = x.view(-1, size)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
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
                image_filename = os.path.join("/home/dai/GPU-Student-2/Cederic/DataSciPro/data/misclsData_LinearCls", f"{global_img_index}_{gt_misclassified}_misclassified.png")

                # Save the image as a PNG file
                image_pil.save(image_filename)

        images = [wandb.Image(image, caption=f"Idx: {str(global_img_index)}, gt: {str(gt_misclassified)}") for image, global_img_index, gt_misclassified in zip(imgs[wrong_indices_per_class], wrong_global_img_indexs, gt_from_misclassified)]
        wandb.log({"misclassified images": images})
        
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
    
    if config.architecture == 'linear':
        model = LinearClassifier(data[0]['img'].shape, num_cls)
        #model.to(config.devices)
    elif config.architecture == 'vqvae':
        #vqvae.to(config.devices)
        model = VQVAEClassifier(num_cls)
        #model.to(config.devices)
    elif config.architecture == 'res50':
        #vqvae.to(config.devices)
        model = ResNet50Classifier(num_cls)
        #model.to(config.devices)
    else:
        print("Sorry, model architecture not implemented!")

    wandb.log({'total_parameters': count_parameters(model)})
    
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    ckpt_callback = ModelCheckpoint(
        dirpath="/home/dai/GPU-Student-2/Cederic/DataSciPro/cls_checkpoints",
        filename="ffhq256." + "b"+ str(config.batch_size) + config.architecture + formatted_time,
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
    
    wandb_config = {
    "run_name": config.run_name,
    "architecture": config.architecture,
    "batch_size": config.batch_size,
    "epochs": config.epochs,
    "lr": config.lr,
    "weight_decay": config.weight_decay,
    "devices": config.devices,
}
    wandb.init(name=config.run_name, project="DataSciPro", entity="cedimac00", config=wandb_config)
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
