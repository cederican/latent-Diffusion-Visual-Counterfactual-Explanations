from diffusers import UNet2DModel, DDIMScheduler, VQModel
import torch
import PIL.Image
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F

from data.dataset import ImageDataset, CelebHQAttrDataset
from init_classifier import Classifier

def color_loss(images, target_color=(1.0, 0.0, 0.0)):
    """Given a target color (R, G, B) return a loss for how far away on average
    the images' pixels are from that color. Defaults to a light teal: (0.1, 0.9, 0.5)"""
    target = torch.tensor(target_color).to(images.device) * 2 - 1  # Map target color to (-1, 1)
    target = target[None, :, None, None]  # Get shape right to work with the images (b, c, h, w)
    error = torch.abs(images - target).mean()  # Mean absolute difference between the image pixels and the target color
    return error

def classifier_loss(classifier, images, targets):
    preds = classifier(images)
    #print(preds[0][31])
    error = torch.abs(preds[0][31] - targets).mean()
    return error

def minDist_loss(images, original_images):
    error = torch.abs(images - original_images).mean()
    return error

#enable checkpointing


# data loading with ground truth no smiling
data = ImageDataset('/home/dai/GPU-Student-2/Cederic/DataSciPro/data/misclsData_gt1', image_size=256, exts=['jpg', 'JPG', 'png'], do_augment=False, sort_names=True)
dataloader = DataLoader(data, batch_size=1, shuffle=False)


#
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "CompVis/ldm-celebahq-256"
seed = 3

# load all models
classifier = Classifier.load_from_checkpoint("/home/dai/GPU-Student-2/Cederic/pjds_group8/cls_checkpoints/ffhq256.2024-05-26 18:46:33.ckpt",
                                             input_dim = data[0]['img'].shape,
                                             num_classes = len(CelebHQAttrDataset.id_to_cls))
classifier.to(device)
classifier.eval()
# check functionality of classifier
all_outputs = []
with torch.no_grad():
    for batch in dataloader:
        inputs = batch['img'].to(classifier.device)
        outputs = classifier(inputs)
        print(outputs[0][31])

        preds_binary = torch.sigmoid(outputs[:, CelebHQAttrDataset.cls_to_id['Smiling']].cpu()) > 0.5
        all_outputs.append(preds_binary) 
all_outputs = torch.cat(all_outputs, dim=0)
print(all_outputs)

unet = UNet2DModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="unet")
vqvae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae")
scheduler = DDIMScheduler.from_config("CompVis/ldm-celebahq-256", subfolder="scheduler")
scheduler.set_timesteps(num_inference_steps=25)

unet.to(device)
vqvae.to(device)

## Inversion
def invert(
    start_latents,
    guidance_scale=3.5,
    num_inference_steps=50,
    device=device,
):

    # Latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(scheduler.timesteps)

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = unet(latent_model_input, t).sample

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = scheduler.alphas_cumprod[current_t]
        alpha_t_next = scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
            1 - alpha_t_next
        ).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)

    return torch.cat(intermediate_latents)


class LatentNoise(torch.nn.Module):
    """
    The LatentNoise Module makes it easier to update the noise tensor with torch optimizers.
    """

    def __init__(self, noise: torch.Tensor):
        super().__init__()
        self.noise = torch.nn.Parameter(noise)

    def forward(self):
        return self.noise


def diffusion_pipe(noise_module: LatentNoise, num_inference_steps):
        z = noise_module()
        for i in tqdm(range(start_step, num_inference_steps)):
            t = scheduler.timesteps[i]
            z = scheduler.scale_model_input(z, t)
            with torch.no_grad():
                noise_pred = unet(z, t)["sample"]
            z = scheduler.step(noise_pred, t, z).prev_sample
        return z

def plot_to_pil(tensor):
    image = tensor.cpu().permute(0, 2, 3, 1).clip(-1,1) * 0.5 + 0.5
    image = PIL.Image.fromarray(np.array(image[0].detach().numpy() * 255).astype(np.uint8))
    plt.imshow(image)
    plt.axis('off')
    plt.show()


# conditional sampling
guidance_cls = 2
guidance_dist = 15
num_inference_steps = 50
num_optimization_steps = 100
start_step = 25


for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    with torch.no_grad():
        z = vqvae.encode(batch['img'].to(device))   # encode the image in the latent space
    z = z.latents
    
    inverted_latents = invert(z, num_inference_steps=50)                  # do the ddim scheduler reversed to add noise to the latents
    z = inverted_latents[-(start_step + 1)].unsqueeze(0)                  # use these latents to start the sampling. better performance when using not the last latent sample
    noise_module = LatentNoise(z.clone()).to(device)                    # convert latent noise to a parameter module for optimization
    noise_module.noise.requires_grad = True
    intermediate_results = []   # list to store the results of the steering

    #new_z = z.clone().detach().requires_grad_(True)

    """
    with torch.no_grad():
        im = vqvae.decode(inverted_latents[-1].unsqueeze(0))
        im_processed = im[0].cpu().permute(0, 2, 3, 1).clip(-1,1) * 0.5 + 0.5
        im_pil = PIL.Image.fromarray(np.array(im_processed[0] * 255).astype(np.uint8))
        plt.imshow(im_pil)
        plt.axis('off')
        plt.show()
    """
    
    optimizer = torch.optim.Adam(
        noise_module.parameters(), lr=0.001, maximize=False # not minimize
    )
    
    """
    
    for i in tqdm(range(start_step, num_inference_steps)):

        t = scheduler.timesteps[i]

        # Prepare the model input
        model_input = scheduler.scale_model_input(x, t)

        # predict noise residual of previous image
        with torch.no_grad():
            noise_pred = unet(model_input, t)["sample"]

        # Set x.requires_grad to True
        x = x.detach().requires_grad_()

        # Get the predicted x0
        x0 = scheduler.step(noise_pred, t, x).pred_original_sample
        decoded_x0 = vqvae.decode(x0)[0]
        decoded_x = vqvae.decode(x)[0]

        # Calculate loss
        #loss = color_loss(x0) * guidance_loss_scale
        loss = guidance_cls * classifier_loss(classifier, decoded_x0, 1.0) + guidance_dist * minDist_loss(decoded_x0, decoded_x)
        if i % 10 == 0:
            print(i, "loss:", loss.item())

        # Get gradient
        cond_grad = -torch.autograd.grad(loss, x)[0]

        # Modify x based on this gradient
        x = x.detach() + cond_grad 

        # Now step with scheduler
        x = scheduler.step(noise_pred, t, x).prev_sample

    # decode image with vae
    with torch.no_grad():
        image = vqvae.decode(x)[0]
    """
    x = torch.zeros_like(z)
    for i in tqdm(range(num_optimization_steps)):
            optimizer.zero_grad()
            x = diffusion_pipe(noise_module, num_inference_steps) # forward
            decoded_x = vqvae.decode(x)[0]

            if i % 10 == 0:
                intermediate_results.append(decoded_x)
                plot_to_pil(decoded_x)

            loss = classifier_loss(classifier, decoded_x, 1.0) # apply losses
            if i % 10 == 0:
                print(i, "loss:", loss.item())
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        image = vqvae.decode(x)[0]
    


    # process image
    image_processed = image.cpu().permute(0, 2, 3, 1).clip(-1,1) * 0.5 + 0.5
    image_pil = PIL.Image.fromarray(np.array(image_processed[0] * 255).astype(np.uint8))
    ori_processed = batch['img'].cpu().permute(0, 2, 3, 1).clip(-1,1) * 0.5 + 0.5
    ori_image = PIL.Image.fromarray(np.array(ori_processed[0] * 255).astype(np.uint8))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(image_pil)
    axs[0].axis('off')
    axs[1].imshow(ori_image)
    axs[1].axis('off')
    plt.show()

    image_pil.save(f"generated_image_{seed}.png")
    print('finish')