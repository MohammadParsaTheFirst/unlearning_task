import torch
import torch.nn as nn
from tqdm import tqdm
from diffusers import UNet2DConditionModel, DDPMScheduler
import torch.nn.functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
from torchvision import transforms
import random


def my_inpainter(model_name: str, dataset_name: str):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output directory
    os.makedirs("Inpainting_Results", exist_ok=True)

    # Load model
    model = UNet2DConditionModel(
        sample_size=64,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(128, 256, 512),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=256,
    ).to(device)

    match model_name:
        case "vanilla":
            unet_path = "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/DDPM/unet_final.pt"
            cl_embedder_path = "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/DDPM/class_embedder.pt"
        case "ssis":
            unet_path = "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/DDPM_Unlearned/unet_unlearned.pt"
            cl_embedder_path = "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/DDPM_Unlearned/class_embedder_unlearned.pt"
        case "dynamic":
            unet_path = "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/DDPM_Unlearned_dynamic/unet_unlearned_vae.pt"
            cl_embedder_path = "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/DDPM_Unlearned_dynamic/class_embedder_unlearned_vae.pt"
        case _:
            unet_path = None
            cl_embedder_path = None

    # Load model states
    try:
        model.load_state_dict(torch.load(unet_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: The {model_name} model weights not found at {unet_path}")
        exit()
    model.eval()

    # Load class embedder
    class_embedder = nn.Embedding(10, 256).to(device)
    try:
        class_embedder.load_state_dict(torch.load(cl_embedder_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: The {model_name} class embedder weights not found at {cl_embedder_path}")
        exit()
    class_embedder.eval()

    # Noise scheduler
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")

    match dataset_name:
        case "mnist":
            dataset_path = "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/Dataset/original_mnist.pt"
        case "augmented":
            dataset_path = "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/Dataset/augmented_mnist_with_trousers.pt"
        case "trouser":
            dataset_path = "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/Dataset/trousers_subset.pt"
        case _:
            dataset_path = None

    # Load the MNIST dataset
    try:
        mnist_data = torch.load(dataset_path)
        images = mnist_data["images"]
        labels = mnist_data["labels"]
    except FileNotFoundError:
        print("Warning: Dataset not found. Using dummy data.")
        images = torch.randn(6000, 28, 28)
        labels = torch.randint(0, 10, (6000,), dtype=torch.long)

    # Randomly select one image
    random_idx = random.randint(0, len(images) - 1)
    selected_image = images[random_idx].unsqueeze(0)  # Shape: (1, 28, 28)
    selected_label = labels[random_idx].item()

    # Resize to 64x64 and normalize
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.Normalize((0.5,), (0.5,))
    ])
    selected_image = selected_image.unsqueeze(0)  # Add channel: (1, 1, 28, 28)
    selected_image = transform(selected_image).to(device)  # Shape: (1, 1, 64, 64)

    # Create a VERTICAL mask for the right half
    mask = torch.ones_like(selected_image)  # 1 for known, 0 for masked
    mask[:, :, :, 32:] = 0  # Mask out right half

    # Prepare masked image: known regions from original, noise in masked regions
    masked_image = selected_image * mask + torch.randn_like(selected_image) * (1 - mask)

    # Inpainting function
    def inpaint(model, class_embedder, scheduler, input_image, mask, condition_class):
        model.eval()
        class_embedder.eval()
        # Initialize with full noise (standard DDPM inpainting starts from noise)
        samples = torch.randn_like(input_image).to(device)
        labels = torch.full((1,), condition_class, dtype=torch.long, device=device)
        embeddings = class_embedder(labels).unsqueeze(1)  # Shape: (1, 1, 256)

        with torch.no_grad():
            for t in tqdm(range(scheduler.config.num_train_timesteps - 1, -1, -1),
                          desc=f"Inpainting class {condition_class}"):
                t_batch = torch.full((1,), t, device=device, dtype=torch.long)

                # Blend the current sample with the input image in unmasked regions
                samples = samples * (1 - mask) + input_image * mask

                # Predict noise
                noise_pred = model(samples, t_batch, encoder_hidden_states=embeddings).sample

                # Perform denoising step
                step_result = scheduler.step(noise_pred, t, samples)
                samples = step_result.prev_sample

        # Final blending to ensure unmasked regions are preserved
        samples = samples * (1 - mask) + input_image * mask
        samples = samples.clamp(-1, 1)  # Clamp to valid range (since normalized to [-1, 1])
        samples_resized = F.interpolate(samples, size=(28, 28), mode='bilinear', align_corners=False)
        return samples_resized

    # Perform inpainting for classes 1 to 9
    plt.figure(figsize=(15, 15))
    plt.subplot(4, 3, 1)
    plt.imshow(selected_image.cpu().squeeze(), cmap='gray')
    plt.title(f"Original Image (Class {selected_label})")
    plt.axis('off')

    plt.subplot(4, 3, 2)
    plt.imshow(masked_image.cpu().squeeze(), cmap='gray')
    plt.title("Masked Input (Right Half)")
    plt.axis('off')

    inpainted_results = []
    for i, condition_class in enumerate(range(1, 10), start=3):
        inpainted_result = inpaint(model, class_embedder, scheduler, masked_image, mask, condition_class)
        inpainted_results.append(inpainted_result)

        plt.subplot(4, 3, i)
        plt.imshow(inpainted_result.cpu().squeeze(), cmap='gray')
        plt.title(f"Inpainted Class {condition_class}")
        plt.axis('off')

        # Save individual inpainted image
        save_image((inpainted_result + 1) / 2,
                   f"Inpainting_Results/{model_name}_inpainted_class{condition_class}_vertical.png")  # Denormalize for saving

    plt.tight_layout()
    plt.savefig(f"Inpainting_Results/{model_name}_inpainting_results_all_classes.png")
    plt.show()

    print("Inpainting results saved to Inpainting_Results/")