import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional

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

class CustomDataset(Dataset):
    """Custom dataset for MNIST-like data with optional transforms."""
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, transform: Optional = None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.images[idx].unsqueeze(0)
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self) -> int:
        return len(self.images)

def load_data(
    retain_path: str,
    forget_path: str,
    batch_size: int = 128,
    transform: Optional = None
) -> Tuple[DataLoader, DataLoader, int, int]:
    """Load retain and forget datasets with error handling."""
    try:
        retain_data = torch.load(retain_path)
        forget_data = torch.load(forget_path)
    except FileNotFoundError:
        print("Warning: Dataset files not found. Using dummy data.")
        retain_data = {"images": torch.randn(60000, 28, 28), "labels": torch.randint(0, 9, (60000,))}
        forget_data = {"images": torch.randn(6000, 28, 28), "labels": torch.ones(6000,).long() * 9}

    # No resize or normalize for VAE (assuming [0,1] range)
    retain_dataset = CustomDataset(retain_data["images"], retain_data["labels"], transform)
    forget_dataset = CustomDataset(forget_data["images"], forget_data["labels"], transform)

    retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    forget_loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return retain_loader, forget_loader, len(retain_dataset), len(forget_dataset)

def initialize_model(
    model_path: str,
    embedder_path: str,
    device: torch.device,
    latent_dim: int = 64,
    num_classes: int = 10,
    embed_dim: int = 256
) -> CVAE:
    """Initialize CVAE model."""
    model = CVAE(latent_dim=latent_dim, num_classes=num_classes, embed_dim=embed_dim).to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded pretrained model weights.")
    except FileNotFoundError:
        print("Warning: Pretrained model weights not found. Using random initialization.")

    # Embedder is part of the model, no separate load needed
    return model

def plot_loss_curves(
    retain_losses: List[float],
    forget_losses: List[float],
    lambda_history: Optional[List[float]] = None,
    title: str = "SISS Unlearning Loss Curves",
    save_path: str = "siss_loss_curves.png"
) -> None:
    """Plot and save loss curves."""
    plt.figure(figsize=(12, 6))
    plt.plot(retain_losses, label="Retain Loss Component")
    plt.plot(forget_losses, label="Forget Loss Component")
    plt.xlabel("Step")
    plt.ylabel("Weighted Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Loss curves saved to {save_path}")
    plt.close()

    if lambda_history:
        plt.figure(figsize=(12, 6))
        plt.plot(lambda_history, label="Lambda (Dynamic)", linestyle='--')
        plt.xlabel("Step")
        plt.ylabel("Lambda")
        plt.title("SISS Unlearning: Lambda")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path.replace("loss_curves", "lambda"))
        print(f"Lambda plot saved to {save_path.replace('loss_curves', 'lambda')}")
        plt.close()

def unlearn_ssis(
    config: Dict,
    retain_loader: DataLoader,
    forget_loader: DataLoader,
    model: CVAE,
    n: int,
    k: int,
    device: torch.device
) -> Tuple[List[float], List[float]]:
    """Perform SISS unlearning for VAE with gradient norm clipping."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    weighted_loss_retain_history = []
    weighted_loss_forget_history = []

    print("Starting SISS unlearning for VAE with gradient norm clipping...")
    step = 0
    for epoch in range(config["epochs"]):
        progress_bar = tqdm(total=min(len(retain_loader), len(forget_loader)))
        progress_bar.set_description(f"Epoch {epoch+1}")

        retain_iter = iter(retain_loader)
        forget_iter = iter(forget_loader)

        for _ in range(min(len(retain_loader), len(forget_loader))):
            x_r, y_r = next(retain_iter)
            x_f, y_f = next(forget_iter)
            x_r, y_r, x_f, y_f = x_r.to(device), y_r.to(device), x_f.to(device), y_f.to(device)

            # Add Gaussian noise
            noise = torch.randn_like(x_r)
            gamma_t = (1 - config["sigma"]**2)**0.5
            sigma = config["sigma"]
            variance_t = sigma**2

            noisy_from_retain = gamma_t * x_r + sigma * noise
            noisy_from_forget = gamma_t * x_f + sigma * noise

            mask = (torch.rand(x_r.size(0), 1, 1, 1, device=device) < config["lambda_"]).float()
            m_t = (1 - mask) * noisy_from_retain + mask * noisy_from_forget
            y_m = ((1 - mask).squeeze() * y_r.float() + mask.squeeze() * y_f.float()).long()

            # Forward pass through VAE
            x_recon, mu, log_var = model(m_t, y_m)

            # Compute reconstruction losses
            bce_x = F.binary_cross_entropy(x_recon, x_r, reduction='none').mean(dim=(1,2,3))
            mse_x = F.mse_loss(x_recon, x_r, reduction='none').mean(dim=(1,2,3))
            recon_x = 0.5 * bce_x + 0.5 * mse_x

            bce_a = F.binary_cross_entropy(x_recon, x_f, reduction='none').mean(dim=(1,2,3))
            mse_a = F.mse_loss(x_recon, x_f, reduction='none').mean(dim=(1,2,3))
            recon_a = 0.5 * bce_a + 0.5 * mse_a

            # Compute KL divergence
            kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)

            # VAE losses per sample
            loss_x = recon_x + config["beta"] * kl
            loss_a = recon_a + config["beta"] * kl

            # Compute weights wx, wa
            dist_sq_x = torch.sum((m_t - gamma_t * x_r)**2, dim=(1,2,3))
            dist_sq_a = torch.sum((m_t - gamma_t * x_f)**2, dim=(1,2,3))
            log_q_x = -0.5 * dist_sq_x / variance_t
            log_q_a = -0.5 * dist_sq_a / variance_t
            log_denom = torch.logsumexp(
                torch.stack([torch.log(1.0 - config["lambda_"]) + log_q_x,
                             torch.log(config["lambda_"]) + log_q_a]),
                dim=0
            )
            wx = torch.exp(log_q_x - log_denom)
            wa = torch.exp(log_q_a - log_denom)

            # Compute terms
            term_x = (n / (n - k)) * (wx * loss_x).mean()
            term_a = (k / (n - k)) * (wa * loss_a).mean()

            # Gradient clipping for term_x and term_a
            optimizer.zero_grad()
            term_x.backward(retain_graph=True)
            grad_x_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))

            optimizer.zero_grad()
            term_a.backward(retain_graph=True)
            grad_a_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))

            scaling_factor = (config["grad_clip_ratio"] * grad_x_norm / (grad_a_norm + 1e-9)).detach()

            # Final loss
            final_loss = term_x - (1 + scaling_factor) * term_a

            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            weighted_loss_retain_history.append(term_x.item())
            weighted_loss_forget_history.append(term_a.item())

            progress_bar.set_postfix({
                "Retain Loss": weighted_loss_retain_history[-1],
                "Forget Loss": weighted_loss_forget_history[-1]
            })
            progress_bar.update(1)
            step += 1

        torch.cuda.empty_cache()

    return weighted_loss_retain_history, weighted_loss_forget_history

class LambdaEncoder(nn.Module):
    """Lambda encoder for dynamic SISS unlearning."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.mu = nn.Linear(32, 1)
        self.logvar = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        lambda_ = torch.sigmoid(z)
        return lambda_, mu, logvar

def unlearn_dynamic(
    config: Dict,
    retain_loader: DataLoader,
    forget_loader: DataLoader,
    model: CVAE,
    n: int,
    k: int,
    device: torch.device
) -> Tuple[List[float], List[float], List[float]]:
    """Perform dynamic SISS unlearning for VAE with learned lambda."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    lambda_encoder = LambdaEncoder().to(device)
    encoder_optimizer = torch.optim.Adam(lambda_encoder.parameters(), lr=config["encoder_lr"])

    lambda_prev = torch.tensor([0.5], device=device)
    weighted_loss_retain_history = []
    weighted_loss_forget_history = []
    lambda_history = []

    print("Starting dynamic SISS unlearning for VAE...")
    step = 0
    for epoch in range(config["epochs"]):
        progress_bar = tqdm(total=min(len(retain_loader), len(forget_loader)))
        progress_bar.set_description(f"Epoch {epoch+1}")

        retain_iter = iter(retain_loader)
        forget_iter = iter(forget_loader)

        for _ in range(min(len(retain_loader), len(forget_loader))):
            x_r, y_r = next(retain_iter)
            x_f, y_f = next(forget_iter)
            x_r, y_r, x_f, y_f = x_r.to(device), y_r.to(device), x_f.to(device), y_f.to(device)

            # Add Gaussian noise
            noise = torch.randn_like(x_r)
            gamma_t = (1 - config["sigma"]**2)**0.5
            sigma = config["sigma"]
            variance_t = sigma**2

            noisy_from_retain = gamma_t * x_r + sigma * noise
            noisy_from_forget = gamma_t * x_f + sigma * noise

            mask = (torch.rand(x_r.size(0), 1, 1, 1, device=device) < lambda_prev.item()).float()
            m_t = (1 - mask) * noisy_from_retain + mask * noisy_from_forget
            y_m = ((1 - mask).squeeze() * y_r.float() + mask.squeeze() * y_f.float()).long()

            # Forward pass through VAE
            x_recon, mu, log_var = model(m_t, y_m)


            """
            CHECK THIS PART VERY CAREFULLY
            """
            # Compute reconstruction losses
            bce_x = F.binary_cross_entropy(x_recon, x_r, reduction='none').mean(dim=(1,2,3))
            mse_x = F.mse_loss(x_recon, x_r, reduction='none').mean(dim=(1,2,3))
            recon_x = 0.5 * bce_x + 0.5 * mse_x

            bce_a = F.binary_cross_entropy(x_recon, x_f, reduction='none').mean(dim=(1,2,3))
            mse_a = F.mse_loss(x_recon, x_f, reduction='none').mean(dim=(1,2,3))
            # i am not sure ! ------------------------------------------------------------------------
            recon_a = 0.5 * bce_a + 0.5 * mse_a

            # Compute KL divergence
            kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)

            # VAE losses per sample
            loss_x = recon_x + config["beta"] * kl
            loss_a = recon_a + config["beta"] * kl

            # Compute weights wx, wa
            dist_sq_x = torch.sum((m_t - gamma_t * x_r)**2, dim=(1,2,3))
            dist_sq_a = torch.sum((m_t - gamma_t * x_f)**2, dim=(1,2,3))
            log_q_x = -0.5 * dist_sq_x / variance_t
            log_q_a = -0.5 * dist_sq_a / variance_t
            log_denom = torch.logsumexp(
                torch.stack([torch.log(1.0 - lambda_prev.item()) + log_q_x,
                             torch.log(lambda_prev.item()) + log_q_a]),
                dim=0
            )
            wx = torch.exp(log_q_x - log_denom)
            wa = torch.exp(log_q_a - log_denom)

            # Compute terms
            term_x = (n / (n - k)) * (wx * loss_x).mean()
            term_a = ((1 + config["forget_scale"]) * (k / (n - k))) * (wa * loss_a).mean()
            siss_loss = term_x - term_a

            # Compute means for encoder input
            loss_x_mean = loss_x.mean()
            loss_a_mean = loss_a.mean()

            # Compute gradients for norms
            retain_grad = torch.autograd.grad(loss_x_mean, model.parameters(), retain_graph=True, allow_unused=True)
            forget_grad = torch.autograd.grad(loss_a_mean, model.parameters(), retain_graph=True, allow_unused=True)

            retain_grad_norm = torch.cat([g.flatten() for g in retain_grad if g is not None]).norm()
            forget_grad_norm = torch.cat([g.flatten() for g in forget_grad if g is not None]).norm()

            # Encoder update
            encoder_input = torch.tensor([
                loss_x_mean.item(),
                loss_a_mean.item(),
                retain_grad_norm.item(),
                forget_grad_norm.item()
            ], device=device).unsqueeze(0)

            lambda_new, mu, logvar = lambda_encoder(encoder_input)
            kl_enc = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            encoder_loss = siss_loss.detach() + config["kl_weight"] * kl_enc

            encoder_optimizer.zero_grad()
            encoder_loss.backward()
            encoder_optimizer.step()

            lambda_prev = lambda_new.detach().squeeze()
            lambda_history.append(lambda_prev.item())

            # Model update
            optimizer.zero_grad()
            siss_loss.backward()
            optimizer.step()

            weighted_loss_retain_history.append(term_x.item())
            weighted_loss_forget_history.append(term_a.item())

            progress_bar.set_postfix({
                "Retain Loss": weighted_loss_retain_history[-1],
                "Forget Loss": weighted_loss_forget_history[-1],
                "Lambda": lambda_prev.item()
            })
            progress_bar.update(1)
            step += 1

        torch.cuda.empty_cache()

    return weighted_loss_retain_history, weighted_loss_forget_history, lambda_history

def unlearn(method_name: str, retain_path: str, forget_path: str) -> None:
    """Main unlearning function to dispatch to specific methods for VAE."""
    config = {
        "epochs": 50,
        "lr": 1e-5,
        "lambda_": 0.5,
        "grad_clip_ratio": 0.1,
        "encoder_lr": 1e-4,
        "forget_scale": 5.0,
        "kl_weight": 0.01,
        "beta": 0.5,
        "sigma": 0.3  # Noise standard deviation for denoising
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    retain_loader, forget_loader, n, k = load_data(retain_path, forget_path)

    model_path = "./models/VAE/vae_final.pt"
    embedder_path = "./models/VAE/class_embedder.pt"  # Not used separately
    model = initialize_model(model_path, embedder_path, device)

    if method_name == "ssis":
        retain_losses, forget_losses = unlearn_ssis(
            config, retain_loader, forget_loader, model, n, k, device
        )
        output_dir = "./models/VAE_Unlearned"
        os.makedirs(output_dir, exist_ok=True)
        save_path = "./Samples/siss_loss_curves_clipped.png"
        title = "SISS Unlearning with Gradient Norm Clipping for VAE"

    elif method_name == "dynamic":
        retain_losses, forget_losses, lambda_history = unlearn_dynamic(
            config, retain_loader, forget_loader, model, n, k, device
        )
        output_dir = "./models/VAE_Unlearned_dynamic"    # state dict of models' params
        os.makedirs(output_dir, exist_ok=True)
        save_path = "./Samples/siss_loss_curves_dynamic.png"       # images/loss curves' paths
        title = "SISS Unlearning: Retain vs. Forget Loss Components for VAE"
    else:
        raise NotImplementedError(f"Method {method_name} not supported.")

    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, f"vae_unlearned{ '_dynamic' if method_name == 'dynamic' else '' }.pt"))
    torch.save(model.class_embedder.state_dict(), os.path.join(output_dir, f"class_embedder_unlearned{ '_dynamic' if method_name == 'dynamic' else '' }.pt"))
    print("Unlearned model and class embedder saved.")

    plot_loss_curves(
        retain_losses,
        forget_losses,
        lambda_history if method_name == "dynamic" else None,
        title=title,
        save_path=save_path
    )

    print("\nUnlearning finished.")

if __name__ == "__main__":
    retain_path = "./data/original_mnist.pt"
    forget_path = "./data/trousers_subset.pt"
    unlearn("ssis", retain_path, forget_path)