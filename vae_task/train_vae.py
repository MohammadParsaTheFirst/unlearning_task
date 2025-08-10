import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.utils import make_grid
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

def train_vae(dataset_name: str, num_epochs=100):
    # --- Step 1: Prepare the dataset ---
    match dataset_name:
        case "augmented":
            DATASET_DIR = "./data/augmented_mnist_with_trousers.pt"
        case "trousers":
            DATASET_DIR = "./data/trousers_subset.pt"
        case "mnist":
            DATASET_DIR = "./data/original_mnist.pt"
        case _:
            raise ValueError(f"Invalid dataset name: {dataset_name}")

    # Load and verify dataset
    data = torch.load(DATASET_DIR)
    images = data["images"]
    labels = data["labels"]
    print(f"Image min: {images.min()}, max: {images.max()}")
    print(f"Labels: {torch.unique(labels)}")

    # Dataset
    class CustomDataset(Dataset):
        def __init__(self, images, labels, transform=None):
            self.images = images / images.max() if images.max() > 1 else images  # Normalize to [0, 1]
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            x = self.images[idx].unsqueeze(0)  # Add channel dimension (1, 28, 28)
            y = self.labels[idx]
            if self.transform:
                x = self.transform(x)
            return x, y

    dataset = CustomDataset(images, labels, transform=None)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Step 2: Define Conditional VAE ---
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
                nn.Conv2d(1 + embed_dim, 32, kernel_size=3, stride=1, padding=1),
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
            self.decoder_input = nn.Linear(latent_dim + embed_dim, 12544)
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
            embed = self.class_embedder(labels).view(-1, self.embed_dim, 1, 1).expand(-1, -1, x.size(2), x.size(3))
            x = torch.cat([x, embed], dim=1)
            h = self.encoder(x)
            mu = self.fc_mu(h)
            log_var = self.fc_log_var(h)
            return mu, log_var

        def reparameterize(self, mu, log_var):
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z, labels):
            embed = self.class_embedder(labels)
            z = torch.cat([z, embed], dim=1)
            h = self.decoder_input(z)
            return self.decoder(h)

        def forward(self, x, labels):
            mu, log_var = self.encode(x, labels)
            z = self.reparameterize(mu, log_var)
            x_recon = self.decode(z, labels)
            return x_recon, mu, log_var

    # --- Step 3: Instantiate model and optimizer ---
    vae_model = CVAE(latent_dim=64, num_classes=10, embed_dim=256).to(device)
    optimizer = torch.optim.AdamW(vae_model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # --- Step 4: Training loop with single progress bar ---
    epochs = num_epochs
    beta = 0.5
    vae_model.train()
    loss_history = []

    with tqdm(total=epochs, desc="Training CVAE", unit="epoch") as pbar:
        for epoch in range(epochs):
            total_loss = 0
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                x_recon, mu, log_var = vae_model(images, labels)

                # Calculate loss
                recon_loss = F.binary_cross_entropy(x_recon, images, reduction="sum")
                mse_loss = F.mse_loss(x_recon, images, reduction="sum")
                recon_loss = 0.5 * recon_loss + 0.5 * mse_loss
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + beta * kl_loss

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / len(dataloader.dataset)
            loss_history.append(avg_loss)
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}"})
            pbar.update(1)

    # --- Step 5: Plot and save loss history ---
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), loss_history, label="Average Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CVAE Training Loss")
    plt.legend()
    plt.grid(True)
    os.makedirs("VAE", exist_ok=True)
    plt.savefig("VAE/loss_plot.png")
    plt.close()
    print(f"Loss plot saved to VAE/loss_plot.png")

    # --- Step 6: Save model and class embedder ---
    model_path = "./models/VAE/vae_final.pt"
    embedder_path = "./models/VAE/class_embedder.pt"
    torch.save(vae_model.state_dict(), model_path)
    torch.save(vae_model.class_embedder.state_dict(), embedder_path)
    print(f"CVAE model weights saved to {model_path}")
    print(f"Class embedder weights saved to {embedder_path}")

    # --- Step 7: Save final samples ---
    vae_model.eval()
    with torch.no_grad():
        sample_labels = torch.randint(0, vae_model.num_classes, (16,), device=device)
        z = torch.randn(16, vae_model.latent_dim).to(device)
        samples = vae_model.decode(z, sample_labels).cpu()
        grid = make_grid(samples, nrow=4, pad_value=1)
        plt.figure(figsize=(8, 8))
        plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
        plt.title("Final CVAE Samples")
        plt.axis("off")
        plt.savefig("VAE/cvae_samples_final.png")
        plt.close()
    print(f"Final samples saved to VAE/cvae_samples_final.png")

if __name__ == "__main__":
    train_vae("augmented")

