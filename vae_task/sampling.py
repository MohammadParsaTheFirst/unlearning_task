import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import os
import glob
import re
from tqdm import tqdm

def sample_vae(model_name: str, index: int):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model paths selection
    match model_name:
        case "vae":
            vae_path = "./models/VAE/vae_final.pt"
            cl_embedder_path = "./models/VAE/class_embedder.pt"
        case _:
            raise ValueError(f"Unknown model_name: {model_name}")

    # Define Conditional VAE model
    class CVAE(nn.Module):
        def __init__(self, latent_dim=64, num_classes=10, embed_dim=256):
            super(CVAE, self).__init__()
            self.latent_dim = latent_dim
            self.num_classes = num_classes
            self.embed_dim = embed_dim

            # Class embedder
            self.class_embedder = nn.Embedding(num_classes, embed_dim)

            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(1 + embed_dim, 32, kernel_size=3, stride=1, padding=1),  # Input: image + embedded label
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten()  # 7x7x256 = 12544
            )

            self.fc_mu = nn.Linear(12544, latent_dim)
            self.fc_log_var = nn.Linear(12544, latent_dim)

            # Decoder
            self.decoder_input = nn.Linear(latent_dim + embed_dim, 12544)  # Include class embedding
            self.decoder = nn.Sequential(
                nn.Unflatten(1, (256, 7, 7)),
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid()
            )

        def encode(self, x, labels):
            # Embed labels and expand to match image dimensions
            embed = self.class_embedder(labels)  # (batch_size, embed_dim)
            embed = embed.view(-1, embed_dim, 1, 1).expand(-1, -1, x.size(2), x.size(3))  # (batch_size, embed_dim, 28, 28)
            x = torch.cat([x, embed], dim=1)  # Concatenate along channel dimension
            h = self.encoder(x)
            mu = self.fc_mu(h)
            log_var = self.fc_log_var(h)
            return mu, log_var

        def reparameterize(self, mu, log_var):
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z, labels):
            # Embed labels and concatenate with latent vector
            embed = self.class_embedder(labels)  # (batch_size, embed_dim)
            z = torch.cat([z, embed], dim=1)  # (batch_size, latent_dim + embed_dim)
            h = self.decoder_input(z)
            return self.decoder(h)

        def forward(self, x, labels):
            mu, log_var = self.encode(x, labels)
            z = self.reparameterize(mu, log_var)
            x_recon = self.decode(z, labels)
            return x_recon, mu, log_var

    # Load model
    vae_model = CVAE(latent_dim=64, num_classes=10, embed_dim=256).to(device)
    if not os.path.isfile(vae_path):
        raise FileNotFoundError(f"VAE checkpoint not found at {vae_path}")
    vae_model.load_state_dict(torch.load(vae_path, map_location=device))
    vae_model.eval()

    # Load class embedder
    class_embedder = nn.Embedding(10, 256).to(device)
    if not os.path.isfile(cl_embedder_path):
        raise FileNotFoundError(f"Class embedder checkpoint not found at {cl_embedder_path}")
    class_embedder.load_state_dict(torch.load(cl_embedder_path, map_location=device))
    class_embedder.eval()

    # Sampling parameters
    condition_class = index  # Class 0–9
    batch_size = 64
    labels = torch.full((batch_size,), condition_class, dtype=torch.long, device=device)

    # Generate samples
    with torch.no_grad():
        # Sample latent vectors
        z = torch.randn(batch_size, vae_model.latent_dim).to(device)
        samples = vae_model.decode(z, labels)

    # Post-processing
    samples = samples.clamp(0, 1)  # Ensure pixel values in [0, 1]

    # Determine save paths
    output_path = f"./vae_task/Samples/class {condition_class}"
    base_name = f"sample_class{condition_class}"
    os.makedirs(output_path, exist_ok=True)

    # Find existing files to avoid overwrite
    pattern = os.path.join(output_path, f"{base_name}_*.png")
    existing_files = glob.glob(pattern)
    max_idx = -1
    regex = re.compile(rf"{re.escape(base_name)}_(\d+)\.png$")
    for f in existing_files:
        m = regex.search(os.path.basename(f))
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
    start_idx = max_idx + 1

    # Save individual images
    for offset, img in enumerate(samples.cpu()):
        idx_to_save = start_idx + offset
        filename = f"{base_name}_{idx_to_save}.png"
        save_image(img, os.path.join(output_path, filename))

    # Save a grid of samples
    grid_img = make_grid(samples.cpu(), nrow=8, normalize=True, pad_value=1)
    plt.figure(figsize=(12, 12))
    plt.axis("off")
    plt.title(f"Generated VAE Samples for Class {condition_class}")
    plt.imshow(grid_img.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.savefig(os.path.join(output_path, f"{base_name}_grid_{start_idx}.png"))
    plt.close()

if __name__ == "__main__":
    sample_vae(model_name="vae", index=1)