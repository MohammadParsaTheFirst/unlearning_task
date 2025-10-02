import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from diffusers import UNet2DConditionModel, DDPMScheduler
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import glob
from typing import Dict, List, Tuple

# Import metric classes from torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image import StructuralSimilarityIndexMeasure

# --- Configuration Constants ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# The number of samples MUST be high (e.g., 10,000) for stable FID/KID/IS calculation.
NUM_SAMPLES = 10000
BATCH_SIZE = 128
IMG_SIZE = 64  # Images are resized to 64x64 in train_diffusion.py


# --- Re-using necessary components from other files ---

class ClassEmbedder(nn.Module):
    """
    Class embedding module used in train_diffusion.py and sampling.py.
    """

    def __init__(self, num_classes: int = 10, embedding_dim: int = 256):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_classes, embedding_dim)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding_layer(labels)


def get_model_paths(model_name: str) -> Tuple[str, str]:
    """
    Defines model and embedder paths based on the model name,
    matching the structure found in sampling.py.
    """
    # NOTE: You may need to adjust these paths based on where your models are saved.
    base_dir = "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist"

    if model_name == "vanilla":
        unet_path = f"{base_dir}/DDPM/unet_final.pt"
        cl_embedder_path = f"{base_dir}/DDPM/class_embedder.pt"
    elif model_name == "ssis":
        unet_path = f"{base_dir}/DDPM_Unlearned/unet_unlearned.pt"
        cl_embedder_path = f"{base_dir}/DDPM_Unlearned/class_embedder_unlearned.pt"
    # Add other model cases (e.g., 'ssis_dynamic') as needed
    elif model_name == "ssis_dynamic":
        unet_path = f"{base_dir}/DDPM_Unlearned_dynamic/unet_unlearned_vae.pt"
        cl_embedder_path = f"{base_dir}/DDPM_Unlearned_dynamic/class_embedder_unlearned_vae.pt"
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return unet_path, cl_embedder_path


def load_models(model_name: str):
    """Loads UNet and ClassEmbedder and moves them to the device."""
    unet_path, cl_embedder_path = get_model_paths(model_name)

    print(f"Loading model '{model_name}' from: {unet_path}")

    # Initialize UNet with parameters matching train_diffusion.py
    model = UNet2DConditionModel(
        sample_size=IMG_SIZE,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(128, 256, 512),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=256,
    ).to(DEVICE)

    # Initialize ClassEmbedder
    class_embedder = ClassEmbedder(num_classes=10, embedding_dim=256).to(DEVICE)

    # Load state dicts
    try:
        model.load_state_dict(torch.load(unet_path, map_location=DEVICE))
        class_embedder.load_state_dict(torch.load(cl_embedder_path, map_location=DEVICE))
        model.eval()
        class_embedder.eval()
    except FileNotFoundError:
        print(f"ERROR: Model files not found for {model_name}. Please check paths.")
        return None, None
    except Exception as e:
        print(f"Error loading model state dicts for {model_name}: {e}")
        return None, None

    return model, class_embedder


@torch.no_grad()
def generate_samples(model, class_embedder, num_samples: int) -> torch.Tensor:
    """
    Generates a large batch of images for evaluation using a standard DDPM sampling loop.
    Returns: Tensor of shape (num_samples, 1, IMG_SIZE, IMG_SIZE), normalized to [-1, 1].
    """
    model.eval()
    class_embedder.eval()

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")

    # Generate labels, balanced across all 10 classes (0-9)
    samples_per_class = num_samples // 10
    labels = torch.cat([
        torch.full((samples_per_class,), i, dtype=torch.long) for i in range(10)
    ], dim=0).to(DEVICE)

    # Adjust total samples if not perfectly divisible
    total_generated = len(labels)

    print(f"Generating {total_generated} samples...")

    generated_images = []

    for i in tqdm(range(0, total_generated, BATCH_SIZE), desc="Sampling"):
        batch_labels = labels[i:i + BATCH_SIZE]
        batch_size = len(batch_labels)

        # Initial noise
        sample = torch.randn(batch_size, 1, IMG_SIZE, IMG_SIZE).to(DEVICE)

        # Get class embeddings
        embeddings = class_embedder(batch_labels).unsqueeze(1)

        # Denoising loop
        for t in noise_scheduler.timesteps:
            # Predict noise residual
            model_output = model(
                sample, t, encoder_hidden_states=embeddings
            ).sample

            # Compute previous noisy sample x_t -> x_{t-1}
            sample = noise_scheduler.step(model_output, t, sample).prev_sample

        generated_images.append(sample.cpu())

    return torch.cat(generated_images, dim=0)


def load_real_data(pt_file: str) -> torch.Tensor:
    """
    Loads and transforms the real dataset to match the generated sample format.
    Returns: Tensor of shape (N, 1, IMG_SIZE, IMG_SIZE), normalized to [-1, 1].
    """
    real_data_path = f"./data/{pt_file}"
    if not os.path.exists(real_data_path):
        print(f"ERROR: Real data file not found at {real_data_path}. Run preprocess_data.py first.")
        return None

    print(f"Loading real data from: {real_data_path}")
    data = torch.load(real_data_path)
    images = data["images"]  # (N, 28, 28)

    # Apply the same transforms used during training
    transform = transforms.Compose([
        transforms.ToDtype(torch.float32, scale=True),  # Ensure float and scale to [0, 1]
        transforms.Resize(IMG_SIZE, antialias=True),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    # Apply transform to all images
    # Add channel dimension (N, 1, H, W)
    images = images.unsqueeze(1)

    transformed_images = []
    for img in tqdm(images, desc="Transforming Real Data"):
        transformed_images.append(transform(img))

    return torch.stack(transformed_images)


def prepare_images_for_inception(images_norm: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Prepares images for FID/IS/KID calculation:
    1. Un-normalize from [-1, 1] to [0, 1].
    2. Convert to 8-bit integers [0, 255].
    3. Repeat channel 1 -> 3.
    4. Subsample the real data to match the number of generated samples (if needed).

    Input: (N, 1, H, W) float in [-1, 1]
    Output: (M, 3, H, W) uint8 in [0, 255], where M <= N
    """

    # 1. Un-normalize and scale to [0, 255] (uint8)
    images_0_255 = ((images_norm.clamp(-1, 1) + 1) / 2) * 255
    images_0_255 = images_0_255.to(torch.uint8)

    # 2. Replicate channel (1 -> 3)
    # The image size must be at least 75x75 for the standard Inception model,
    # but TorchMetrics automatically handles resizing to 299x299 internally.
    images_rgb = images_0_255.repeat(1, 3, 1, 1)

    # 3. Subsample (only for real images if needed)
    if images_rgb.shape[0] > num_samples:
        # Use a consistent slice to ensure deterministic metric results
        images_rgb = images_rgb[:num_samples]

    return images_rgb


def calculate_metrics(real_images_norm: torch.Tensor, fake_images_norm: torch.Tensor):
    """
    Calculates FID, KID, IS, and SSIM between the two datasets.
    """
    print("\nStarting metric calculation...")

    # Prepare FID, KID, IS: Requires uint8, 3-channel, 299x299 (TorchMetrics resizes)
    real_images_inception = prepare_images_for_inception(real_images_norm, len(fake_images_norm))
    fake_images_inception = prepare_images_for_inception(fake_images_norm, len(fake_images_norm))

    # --- FID ---
    # `feature=2048` uses the features from the final average pooling layer of Inception V3
    fid = FrechetInceptionDistance(feature=2048, reset_real_features=False).to(DEVICE)

    # Move to device and update
    fid.update(real_images_inception.to(DEVICE), real=True)
    fid.update(fake_images_inception.to(DEVICE), real=False)
    fid_score = fid.compute().item()

    # --- KID ---
    # `subsets=10` and `subset_size=1000` are common settings for better stability
    kid = KernelInceptionDistance(subsets=10, subset_size=1000, feature=2048, reset_real_features=False).to(DEVICE)

    kid.update(real_images_inception.to(DEVICE), real=True)
    kid.update(fake_images_inception.to(DEVICE), real=False)
    kid_mean, kid_std = kid.compute()

    # --- IS ---
    # Inception Score requires the full 299x299 input for internal resizing
    # Note: IS is generally less reliable than FID/KID for low-resolution, non-ImageNet data
    is_metric = InceptionScore(splits=10, normalize_input=False).to(DEVICE)
    is_metric.update(fake_images_inception.to(DEVICE))
    is_mean, is_std = is_metric.compute()

    # --- SSIM ---
    # SSIM is calculated between the two datasets (real vs fake, on the first channel)
    # Note: SSIM is better for comparing *reconstructions* or small differences,
    # not usually for entire generated distribution.
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='mean').to(DEVICE)

    # Calculate SSIM only for the generated samples against the *first* real image
    # as a simple reference (a more complex SSIM distribution comparison is not standard).
    # A proper SSIM comparison requires matching pairs of real/fake images, which is not
    # the case here. We will calculate the average SSIM of all generated images
    # against the mean real image for a proxy score.

    # Compute mean real image
    mean_real_image = real_images_norm.mean(dim=0, keepdim=True).to(DEVICE)  # (1, 1, 64, 64)
    # Repeat the mean image for all generated samples
    mean_real_batch = mean_real_image.repeat(len(fake_images_norm), 1, 1, 1)  # (N, 1, 64, 64)

    # Calculate SSIM between generated and mean real image
    ssim_score = ssim(fake_images_norm.to(DEVICE), mean_real_batch).item()

    results = {
        "FID": fid_score,
        "KID_Mean": kid_mean.item(),
        "KID_Std": kid_std.item(),
        "IS_Mean": is_mean.item(),
        "IS_Std": is_std.item(),
        "SSIM_vs_Mean_Real": ssim_score,
    }

    return results


def main():
    """Main function to run the evaluation pipeline."""

    # 1. Define models to evaluate
    model_names = ["vanilla", "ssis", "ssis_dynamic"]

    # 2. Load and prepare real data (only once)
    real_data_pt_file = "augmented_mnist_with_trousers.pt"  # Check train_diffusion.py and preprocess_data.py
    real_images_norm = load_real_data(real_data_pt_file)  # (N, 1, 64, 64) in [-1, 1]

    if real_images_norm is None:
        return

    print(f"Total real images loaded: {len(real_images_norm)}")

    all_results = {}

    for name in model_names:
        print("-" * 50)
        print(f"--- Evaluating Model: {name} ---")

        # 3. Load Model
        model, class_embedder = load_models(name)

        if model is None:
            continue

        # 4. Generate Samples
        fake_images_norm = generate_samples(model, class_embedder, NUM_SAMPLES)  # (NUM_SAMPLES, 1, 64, 64) in [-1, 1]

        # 5. Calculate Metrics
        results = calculate_metrics(real_images_norm, fake_images_norm)
        all_results[name] = results

        # 6. Print Results
        print("\n--- Evaluation Summary ---")
        for metric, value in results.items():
            if "_Std" in metric:
                # Combine mean and std for KID and IS
                mean_key = metric.replace("_Std", "_Mean")
                if mean_key not in results:
                    continue
                mean_val = results[mean_key]
                print(f"  {mean_key.replace('_Mean', '')}: {mean_val:.4f} \u00B1 {value:.4f}")
            elif "_Mean" not in metric and metric != "FID":
                print(f"  {metric}: {value:.4f}")
        print(f"  FID: {results['FID']:.4f}")
        print("-" * 50)

    print("\n\n--- FINAL COMPARISON ---")
    header = "Model Name".ljust(15) + " | " + "FID".ljust(8) + " | " + "KID (\u00B1)".ljust(
        12) + " | " + "IS (\u00B1)".ljust(12) + " | " + "SSIM".ljust(8)
    print(header)
    print("-" * len(header))
    for name, res in all_results.items():
        kid_str = f"{res['KID_Mean']:.3f} \u00B1 {res['KID_Std']:.3f}"
        is_str = f"{res['IS_Mean']:.3f} \u00B1 {res['IS_Std']:.3f}"
        row = (
                name.ljust(15) + " | " +
                f"{res['FID']:.4f}".ljust(8) + " | " +
                kid_str.ljust(12) + " | " +
                is_str.ljust(12) + " | " +
                f"{res['SSIM_vs_Mean_Real']:.4f}".ljust(8)
        )
        print(row)

    print("-" * len(header))


if __name__ == "__main__":
    main()