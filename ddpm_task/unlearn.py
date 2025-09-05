import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from diffusers import UNet2DConditionModel, DDPMScheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional


class CustomDataset(Dataset):
    """Custom dataset for MNIST-like data with optional transforms."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, transform: Optional[transforms.Compose] = None):
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
        transform: Optional[transforms.Compose] = None
) -> Tuple[DataLoader, DataLoader, int, int]:
    """Load retain and forget datasets with error handling."""
    try:
        retain_data = torch.load(retain_path)
        forget_data = torch.load(forget_path)
    except FileNotFoundError:
        print("Warning: Dataset files not found. Using dummy data.")
        retain_data = {"images": torch.randn(60000, 28, 28), "labels": torch.randint(0, 9, (60000,))}
        forget_data = {"images": torch.randn(6000, 28, 28), "labels": torch.ones(6000, ).long() * 9}

    transform = transform or transforms.Compose([
        transforms.Resize(64),
        transforms.Normalize((0.5,), (0.5,))
    ])

    retain_dataset = CustomDataset(retain_data["images"], retain_data["labels"], transform)
    forget_dataset = CustomDataset(forget_data["images"], forget_data["labels"], transform)

    retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    forget_loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return retain_loader, forget_loader, len(retain_dataset), len(forget_dataset)


def initialize_model(
        model_path: str,
        embedder_path: str,
        device: torch.device,
        embedding_dim: int = 256
) -> Tuple[UNet2DConditionModel, nn.Embedding]:
    """Initialize UNet and class embedder models."""
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

    class_embedder = nn.Embedding(10, embedding_dim).to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        class_embedder.load_state_dict(torch.load(embedder_path, map_location=device))
        print("Loaded pretrained model weights.")
    except FileNotFoundError:
        print("Warning: Pretrained model weights not found. Using random initialization.")

    return model, class_embedder


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
    plt.ylabel("Weighted MSE Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Loss curves saved to {save_path}")
    plt.show()

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
        plt.show()


def unlearn_ssis(
        config: Dict,
        retain_loader: DataLoader,
        forget_loader: DataLoader,
        model: UNet2DConditionModel,
        class_embedder: nn.Embedding,
        n: int,
        k: int,
        device: torch.device
) -> Tuple[List[float], List[float]]:
    """Perform SISS unlearning with gradient norm clipping."""
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(class_embedder.parameters()),
        lr=config["lr"]
    )

    model.train()
    weighted_loss_retain_history = []
    weighted_loss_forget_history = []

    print("Starting SISS unlearning with gradient norm clipping...")
    for epoch in range(config["epochs"]):
        progress_bar = tqdm(total=min(len(retain_loader), len(forget_loader)))
        progress_bar.set_description(f"Epoch {epoch + 1}")

        retain_iter = iter(retain_loader)
        forget_iter = iter(forget_loader)

        for _ in range(min(len(retain_loader), len(forget_loader))):
            x_r, y_r = next(retain_iter)
            x_f, y_f = next(forget_iter)
            x_r, y_r, x_f, y_f = x_r.to(device), y_r.to(device), x_f.to(device), y_f.to(device)

            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (x_r.size(0),),
                                      device=device).long()
            noise = torch.randn_like(x_r)
            noisy_from_retain = noise_scheduler.add_noise(x_r, noise, timesteps)
            noisy_from_forget = noise_scheduler.add_noise(x_f, noise, timesteps)
            mask = (torch.rand(x_r.size(0), 1, 1, 1, device=device) < config["lambda_"]).float()
            m_t = (1 - mask) * noisy_from_retain + mask * noisy_from_forget
            y_m = ((1 - mask.squeeze()) * y_r.float() + mask.squeeze() * y_f.float()).long()

            emb_m = class_embedder(y_m).unsqueeze(1)
            noise_pred = model(m_t, timesteps, encoder_hidden_states=emb_m).sample

            alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
            sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[timesteps]).view(-1, 1, 1, 1)
            sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alphas_cumprod[timesteps]).view(-1, 1, 1, 1)
            variance_t = (sqrt_one_minus_alphas_cumprod_t ** 2).view(-1)
            dist_sq_x = torch.sum((m_t - sqrt_alphas_cumprod_t * x_r) ** 2, dim=(1, 2, 3))
            dist_sq_a = torch.sum((m_t - sqrt_alphas_cumprod_t * x_f) ** 2, dim=(1, 2, 3))
            log_q_x = -0.5 * dist_sq_x / variance_t
            log_q_a = -0.5 * dist_sq_a / variance_t
            log_denom = torch.logsumexp(
                torch.stack([torch.log(torch.tensor(1.0 - config["lambda_"])) + log_q_x,
                             torch.log(torch.tensor(config["lambda_"])) + log_q_a]),
                dim=0
            )
            wx = torch.exp(log_q_x - log_denom)
            wa = torch.exp(log_q_a - log_denom)

            gt_noise_x = (m_t - sqrt_alphas_cumprod_t * x_r) / sqrt_one_minus_alphas_cumprod_t
            gt_noise_a = (m_t - sqrt_alphas_cumprod_t * x_f) / sqrt_one_minus_alphas_cumprod_t
            loss_x = nn.functional.mse_loss(noise_pred, gt_noise_x, reduction='none').mean(dim=(1, 2, 3))
            loss_a = nn.functional.mse_loss(noise_pred, gt_noise_a, reduction='none').mean(dim=(1, 2, 3))

            term_x = ((n / (n - k)) * wx * loss_x).mean()
            term_a = ((k / (n - k)) * wa * loss_a).mean()

            optimizer.zero_grad()
            term_x.backward(retain_graph=True)
            grad_x_norm = torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(class_embedder.parameters()),
                max_norm=float('inf')
            )

            optimizer.zero_grad()
            term_a.backward(retain_graph=True)
            grad_a_norm = torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(class_embedder.parameters()),
                max_norm=float('inf')
            )

            scaling_factor = ((config["grad_clip_ratio"] * grad_x_norm) / (grad_a_norm + 1e-9)).detach()
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
        model: UNet2DConditionModel,
        class_embedder: nn.Embedding,
        n: int,
        k: int,
        device: torch.device
) -> Tuple[List[float], List[float], List[float]]:
    """Perform dynamic SISS unlearning with learned lambda."""
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(class_embedder.parameters()),
        lr=config["lr"]
    )
    lambda_encoder = LambdaEncoder().to(device)
    encoder_optimizer = torch.optim.Adam(lambda_encoder.parameters(), lr=config["encoder_lr"])

    lambda_prev = torch.tensor([0.5], device=device)
    model.train()
    weighted_loss_retain_history = []
    weighted_loss_forget_history = []
    lambda_history = []

    print("Starting dynamic SISS unlearning...")
    for epoch in range(config["epochs"]):
        progress_bar = tqdm(total=min(len(retain_loader), len(forget_loader)))
        progress_bar.set_description(f"Epoch {epoch + 1}")

        retain_iter = iter(retain_loader)
        forget_iter = iter(forget_loader)

        for _ in range(min(len(retain_loader), len(forget_loader))):
            x_r, y_r = next(retain_iter)
            x_f, y_f = next(forget_iter)
            x_r, y_r, x_f, y_f = x_r.to(device), y_r.to(device), x_f.to(device), y_f.to(device)

            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (x_r.size(0),),
                                      device=device).long()
            noise = torch.randn_like(x_r)
            noisy_from_retain = noise_scheduler.add_noise(x_r, noise, timesteps)
            noisy_from_forget = noise_scheduler.add_noise(x_f, noise, timesteps)
            mask = (torch.rand(x_r.size(0), 1, 1, 1, device=device) < lambda_prev.item()).float()
            m_t = (1 - mask) * noisy_from_retain + mask * noisy_from_forget
            y_m = ((1 - mask.squeeze()) * y_r.float() + mask.squeeze() * y_f.float()).long()

            emb_m = class_embedder(y_m).unsqueeze(1)
            noise_pred = model(m_t, timesteps, encoder_hidden_states=emb_m).sample

            alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
            sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[timesteps]).view(-1, 1, 1, 1)
            sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alphas_cumprod[timesteps]).view(-1, 1, 1, 1)
            variance_t = (sqrt_one_minus_alphas_cumprod_t ** 2).view(-1)
            dist_sq_x = torch.sum((m_t - sqrt_alphas_cumprod_t * x_r) ** 2, dim=(1, 2, 3))
            dist_sq_a = torch.sum((m_t - sqrt_alphas_cumprod_t * x_f) ** 2, dim=(1, 2, 3))
            log_q_x = -0.5 * dist_sq_x / variance_t
            log_q_a = -0.5 * dist_sq_a / variance_t
            log_denom = torch.logsumexp(
                torch.stack([
                    torch.log(torch.tensor(1.0 - lambda_prev.item())) + log_q_x,
                    torch.log(torch.tensor(lambda_prev.item())) + log_q_a
                ]), dim=0
            )
            wx = torch.exp(log_q_x - log_denom)
            wa = torch.exp(log_q_a - log_denom)

            gt_noise_x = (m_t - sqrt_alphas_cumprod_t * x_r) / sqrt_one_minus_alphas_cumprod_t
            gt_noise_a = (m_t - sqrt_alphas_cumprod_t * x_f) / sqrt_one_minus_alphas_cumprod_t
            loss_x = nn.functional.mse_loss(noise_pred, gt_noise_x, reduction='none').mean(dim=(1, 2, 3))
            loss_a = nn.functional.mse_loss(noise_pred, gt_noise_a, reduction='none').mean(dim=(1, 2, 3))

            term_x = (n / (n - k)) * wx * loss_x
            term_a = ((1 + config["forget_scale"]) * k / (n - k)) * wa * loss_a
            siss_loss = (term_x - term_a).mean()

            loss_x_mean = loss_x.mean()
            loss_a_mean = loss_a.mean()

            retain_grad = torch.autograd.grad(loss_x_mean, model.parameters(), retain_graph=True, allow_unused=True)
            forget_grad = torch.autograd.grad(loss_a_mean, model.parameters(), retain_graph=True, allow_unused=True)

            retain_grad_norm = torch.cat([g.flatten() for g in retain_grad if g is not None]).norm()
            forget_grad_norm = torch.cat([g.flatten() for g in forget_grad if g is not None]).norm()

            encoder_input = torch.tensor([
                loss_x_mean.item(),
                loss_a_mean.item(),
                retain_grad_norm.item(),
                forget_grad_norm.item()
            ], device=device).unsqueeze(0)

            lambda_new, mu, logvar = lambda_encoder(encoder_input)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            encoder_loss = siss_loss.detach() + config["kl_weight"] * kl

            encoder_optimizer.zero_grad()
            encoder_loss.backward()
            encoder_optimizer.step()

            lambda_prev = lambda_new.detach().squeeze()
            lambda_history.append(lambda_prev.item())

            optimizer.zero_grad()
            siss_loss.backward()
            optimizer.step()

            weighted_loss_retain_history.append(term_x.mean().item())
            weighted_loss_forget_history.append(term_a.mean().item())

            progress_bar.set_postfix({
                "Retain Loss": weighted_loss_retain_history[-1],
                "Forget Loss": weighted_loss_forget_history[-1],
                "Lambda": lambda_prev.item()
            })
            progress_bar.update(1)

        torch.cuda.empty_cache()

    return weighted_loss_retain_history, weighted_loss_forget_history, lambda_history


def unlearn(method_name: str, retain_path: str, forget_path: str, model_name: str) -> None:
    """Main unlearning function to dispatch to specific methods."""
    config = {
        "epochs": 50,
        "lr": 1e-5,
        "lambda_": 0.5,
        "grad_clip_ratio": 0.1,
        "encoder_lr": 1e-4,
        "forget_scale": 5.0,
        "kl_weight": 0.01
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    retain_loader, forget_loader, n, k = load_data(retain_path, forget_path)

    model_path = "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/DDPM/unet_final.pt"
    embedder_path = "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/DDPM/class_embedder.pt"
    model, class_embedder = initialize_model(model_path, embedder_path, device)

    if method_name == "ssis":
        retain_losses, forget_losses = unlearn_ssis(
            config, retain_loader, forget_loader, model, class_embedder, n, k, device
        )
        output_dir = "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/DDPM_Unlearned"
        save_path = "siss_loss_curves_clipped.png"
        title = "SISS Unlearning with Gradient Norm Clipping"

    elif method_name == "dynamic":
        retain_losses, forget_losses, lambda_history = unlearn_dynamic(
            config, retain_loader, forget_loader, model, class_embedder, n, k, device
        )
        output_dir = "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/DDPM_Unlearned_dynamic"
        save_path = "siss_loss_curves_dynamic.png"
        title = "SISS Unlearning: Retain vs. Forget Loss Components"
    else:
        raise NotImplementedError(f"Method {method_name} not supported.")

    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(output_dir, f"unet_unlearned{'_vae' if method_name == 'dynamic' else ''}.pt"))
    torch.save(class_embedder.state_dict(),
               os.path.join(output_dir, f"class_embedder_unlearned{'_vae' if method_name == 'dynamic' else ''}.pt"))
    print("Unlearned model and class embedder saved.")

    plot_loss_curves(
        retain_losses,
        forget_losses,
        lambda_history if method_name == "dynamic" else None,
        title=title,
        save_path=save_path
    )

    print("\nUnlearning finished.")


# ------------------------------------------------------------------
# if __name__ == "__main__":
#     # Configuration for SSIS unlearning
#     retain_path = "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/Dataset/original_mnist.pt"
#     forget_path = "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/Dataset/trousers_subset.pt"
#     model_name = "unet_final.pt"

#     # Run SSIS unlearning
#     try:
#         unlearn(
#             method_name="ssis",
#             retain_path=retain_path,
#             forget_path=forget_path,
#             model_name=model_name
#         )
#     except Exception as e:
#         print(f"Error during SSIS unlearning: {e}")
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Configuration for dynamic unlearning
    retain_path = "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/Dataset/original_mnist.pt"
    forget_path = "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/Dataset/trousers_subset.pt"
    model_name = "unet_final.pt"

    # Run dynamic unlearning
    try:
        unlearn(
            method_name="dynamic",
            retain_path=retain_path,
            forget_path=forget_path,
            model_name=model_name
        )
    except Exception as e:
        print(f"Error during dynamic unlearning: {e}")
# ----------------------------------------------------------------------


# import torch
# from torch import nn
# from torch.utils.data import DataLoader,  Subset
# from torchvision import transforms
# from diffusers import UNet2DConditionModel, DDPMScheduler
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import os

# def unlearn(method_name: str, retain_path: str, forget_path: str, model_name:str ):
#     if method_name == "ssis":
#         # --- (Dataset and Model setup from your code remains the same) ---
#         # Load retain (original MNIST) and forget (trousers only) datasets
#         try:
#             retain_data = torch.load("/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/Dataset/original_mnist.pt")
#             forget_data = torch.load("/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/Dataset/trousers_subset.pt")
#         except FileNotFoundError:
#             print("Warning: Dataset files not found. Using dummy data.")
#             retain_data = {"images": torch.randn(60000, 28, 28), "labels": torch.randint(0, 9, (60000,))}
#             forget_data = {"images": torch.randn(6000, 28, 28), "labels": torch.ones(6000,).long() * 9}

#         transform = transforms.Compose([
#             transforms.Resize(64),
#             transforms.Normalize((0.5,), (0.5,))
#         ])

#         class CustomDataset(torch.utils.data.Dataset):
#             def __init__(self, images, labels, transform=None):
#                 self.images = images
#                 self.labels = labels
#                 self.transform = transform

#             def __getitem__(self, idx):
#                 x = self.images[idx].unsqueeze(0)
#                 y = self.labels[idx]
#                 if self.transform:
#                     x = self.transform(x)
#                 return x, y

#             def __len__(self):
#                 return len(self.images)

#         retain_dataset = CustomDataset(retain_data["images"], retain_data["labels"], transform)
#         forget_dataset = CustomDataset(forget_data["images"], forget_data["labels"], transform)

#         retain_loader = DataLoader(retain_dataset, batch_size=128, shuffle=True, drop_last=True)
#         forget_loader = DataLoader(forget_dataset, batch_size=128, shuffle=True, drop_last=True)


#         # Model and embedder
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         model = UNet2DConditionModel(
#             sample_size=64, in_channels=1, out_channels=1, layers_per_block=2,
#             block_out_channels=(128, 256, 512),
#             down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
#             up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
#             cross_attention_dim=256,
#         ).to(device)

#         embedding_dim = 256
#         class_embedder = nn.Embedding(10, embedding_dim).to(device)

#         try:
#             model.load_state_dict(torch.load("/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/DDPM/unet_final.pt", map_location=device))
#             class_embedder.load_state_dict(torch.load("work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/DDPM/class_embedder.pt", map_location=device))
#             print("Loaded pretrained model weights.")
#         except FileNotFoundError:
#             print("Warning: Pretrained model weights not found. Using random initialization.")

#         noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
#         optimizer = torch.optim.AdamW(list(model.parameters()) + list(class_embedder.parameters()), lr=1e-5)

#         # SISS unlearning parameters
#         lambda_ = 0.5
#         grad_clip_ratio = 0.1 # Ratio for clipping the forget grad norm (10% in the paper)
#         k = len(forget_dataset)
#         n = len(retain_dataset) + k

#         # History trackers for plotting
#         weighted_loss_retain_history = []
#         weighted_loss_forget_history = []

#         model.train()
#         epochs = 50

#         print("Starting SISS unlearning with gradient norm clipping...")
#         for epoch in range(epochs):
#             progress_bar = tqdm(total=min(len(retain_loader), len(forget_loader)))
#             progress_bar.set_description(f"Epoch {epoch+1}")

#             retain_iter = iter(retain_loader)
#             forget_iter = iter(forget_loader)

#             for _ in range(min(len(retain_loader), len(forget_loader))):
#                 x_r, y_r = next(retain_iter)
#                 x_f, y_f = next(forget_iter)

#                 x_r, y_r = x_r.to(device), y_r.to(device)
#                 x_f, y_f = x_f.to(device), y_f.to(device)

#                 # 1. & 2. Mixture Sampling and Forward Pass
#                 timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (x_r.size(0),), device=device).long()
#                 noise = torch.randn_like(x_r)
#                 noisy_from_retain = noise_scheduler.add_noise(x_r, noise, timesteps)
#                 noisy_from_forget = noise_scheduler.add_noise(x_f, noise, timesteps)
#                 mask = (torch.rand(x_r.size(0), 1, 1, 1, device=device) < lambda_).float()
#                 m_t = (1 - mask) * noisy_from_retain + mask * noisy_from_forget
#                 y_m = ((1 - mask.squeeze()) * y_r.float() + mask.squeeze() * y_f.float()).long()

#                 emb_m = class_embedder(y_m).unsqueeze(1)
#                 noise_pred = model(m_t, timesteps, encoder_hidden_states=emb_m).sample

#                 # 3. Calculate Importance Weights
#                 alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
#                 sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[timesteps]).view(-1, 1, 1, 1)
#                 sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alphas_cumprod[timesteps]).view(-1, 1, 1, 1)
#                 variance_t = (sqrt_one_minus_alphas_cumprod_t**2).view(-1)
#                 dist_sq_x = torch.sum((m_t - sqrt_alphas_cumprod_t * x_r)**2, dim=(1,2,3))
#                 dist_sq_a = torch.sum((m_t - sqrt_alphas_cumprod_t * x_f)**2, dim=(1,2,3))
#                 log_q_x = -0.5 * dist_sq_x / variance_t
#                 log_q_a = -0.5 * dist_sq_a / variance_t
#                 log_denom = torch.logsumexp(
#                     torch.stack([torch.log(torch.tensor(1.0 - lambda_)) + log_q_x, torch.log(torch.tensor(lambda_)) + log_q_a]),
#                     dim=0
#                 )
#                 wx = torch.exp(log_q_x - log_denom)
#                 wa = torch.exp(log_q_a - log_denom)

#                 # 4. Calculate SISS loss components
#                 gt_noise_x = (m_t - sqrt_alphas_cumprod_t * x_r) / sqrt_one_minus_alphas_cumprod_t
#                 gt_noise_a = (m_t - sqrt_alphas_cumprod_t * x_f) / sqrt_one_minus_alphas_cumprod_t
#                 loss_x = nn.functional.mse_loss(noise_pred, gt_noise_x, reduction='none').mean(dim=(1,2,3))
#                 loss_a = nn.functional.mse_loss(noise_pred, gt_noise_a, reduction='none').mean(dim=(1,2,3))

#                 term_x = ((n / (n - k)) * wx * loss_x).mean()
#                 term_a = ((k / (n - k)) * wa * loss_a).mean()

#                 # 5. GRADIENT NORM CLIPPING
#                 optimizer.zero_grad()

#                 # Calculate gradients for the retain term, keeping the graph
#                 term_x.backward(retain_graph=True)
#                 grad_x_norm = torch.nn.utils.clip_grad_norm_(
#                     list(model.parameters()) + list(class_embedder.parameters()),
#                     max_norm=float('inf')
#                 )

#                 optimizer.zero_grad()

#                 # Calculate gradients for the forget term, ALSO keeping the graph
#                 # This is the key fix to prevent the runtime error.
#                 term_a.backward(retain_graph=True)
#                 grad_a_norm = torch.nn.utils.clip_grad_norm_(
#                     list(model.parameters()) + list(class_embedder.parameters()),
#                     max_norm=float('inf')
#                 )

#                 # 6. DYNAMICALLY SCALE AND COMBINE
#                 # Detach the scaling factor from the graph to treat it as a constant
#                 scaling_factor = ((grad_clip_ratio * grad_x_norm) / (grad_a_norm + 1e-9)).detach()

#                 # Calculate the final combined loss for the actual optimizer step
#                 final_loss = term_x - (1+scaling_factor) * term_a

#                 # 7. FINAL OPTIMIZATION STEP
#                 # We need to calculate the gradients for this final combined loss.
#                 # Since we've been manipulating grads, it's cleanest to zero them out
#                 # and do one final backward pass.
#                 optimizer.zero_grad()
#                 final_loss.backward()
#                 optimizer.step()

#                 # Log the weighted loss components for plotting
#                 weighted_loss_retain_history.append(term_x.item())
#                 weighted_loss_forget_history.append((term_a).item())

#                 progress_bar.set_postfix({
#                     "Retain Loss": weighted_loss_retain_history[-1],
#                     "Forget Loss": weighted_loss_forget_history[-1]
#                 })
#                 progress_bar.update(1)

#             torch.cuda.empty_cache()

#         print("\nUnlearning finished.")

#         # Save the unlearned model
#         os.makedirs("/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/DDPM_Unlearned", exist_ok=True)
#         torch.save(model.state_dict(), "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/DDPM_Unlearned/unet_unlearned.pt")
#         torch.save(class_embedder.state_dict(), "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/DDPM_Unlearned/class_embedder_unlearned.pt")
#         print("Unlearned model and class embedder saved.")

#         # PLOTTING
#         plt.figure(figsize=(12, 6))
#         plt.plot(weighted_loss_retain_history, label="Weighted Retain Loss Component")
#         plt.plot(weighted_loss_forget_history, label="Dynamically Scaled Forget Loss Component")
#         plt.xlabel("Step")
#         plt.ylabel("Weighted MSE Loss")
#         plt.title("SISS Unlearning with Gradient Norm Clipping")
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig("siss_loss_curves_clipped.png")
#         print("Loss curves saved to siss_loss_curves_clipped.png")
#         plt.show()

#     elif method_name == "dynamic":
#         # --- (Dataset and Model setup from your code remains the same) ---

#         retain_path = "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/Dataset/original_mnist.pt"
#         forget_path = "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/Dataset/trousers_subset.pt"

#         try:
#             retain_data = torch.load(retain_path)
#             forget_data = torch.load(forget_path)
#         except FileNotFoundError:
#             print("Warning: Dataset files not found. Using dummy data.")
#             retain_data = {"images": torch.randn(60000, 28, 28), "labels": torch.randint(0, 9, (60000,))}
#             forget_data = {"images": torch.randn(6000, 28, 28), "labels": torch.ones(6000,).long() * 9}

#         transform = transforms.Compose([
#             transforms.Resize(64),
#             transforms.Normalize((0.5,), (0.5,))
#         ])

#         class CustomDataset(torch.utils.data.Dataset):
#             def __init__(self, images, labels, transform=None):
#                 self.images = images
#                 self.labels = labels
#                 self.transform = transform

#             def __getitem__(self, idx):
#                 x = self.images[idx].unsqueeze(0)
#                 y = self.labels[idx]
#                 if self.transform:
#                     x = self.transform(x)
#                 return x, y

#             def __len__(self):
#                 return len(self.images)

#         retain_dataset = CustomDataset(retain_data["images"], retain_data["labels"], transform)
#         forget_dataset = CustomDataset(forget_data["images"], forget_data["labels"], transform)

#         retain_loader = DataLoader(retain_dataset, batch_size=128, shuffle=True, drop_last=True)
#         forget_loader = DataLoader(forget_dataset, batch_size=128, shuffle=True, drop_last=True)

#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         model = UNet2DConditionModel(
#             sample_size=64, in_channels=1, out_channels=1, layers_per_block=2,
#             block_out_channels=(128, 256, 512),
#             down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
#             up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
#             cross_attention_dim=256,
#         ).to(device)

#         embedding_dim = 256
#         class_embedder = nn.Embedding(10, embedding_dim).to(device)

#         unet_path = "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/DDPM/unet_final.pt"
#         cl_embedd_path = "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/DDPM/class_embedder.pt"
#         try:
#             model.load_state_dict(torch.load(unet_path, map_location=device))
#             class_embedder.load_state_dict(torch.load(cl_embedd_path, map_location=device))
#             print("Loaded pretrained model weights.")
#         except FileNotFoundError:
#             print("Warning: Pretrained model weights not found. Using random initialization.")

#         noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
#         optimizer = torch.optim.AdamW(list(model.parameters()) + list(class_embedder.parameters()), lr=1e-5)

#         # Lambda encoder definition
#         class LambdaEncoder(nn.Module):
#             def __init__(self):
#                 super().__init__()
#                 self.net = nn.Sequential(
#                     nn.Linear(4, 64),
#                     nn.ReLU(),
#                     nn.Linear(64, 32),
#                     nn.ReLU()
#                 )
#                 self.mu = nn.Linear(32, 1)
#                 self.logvar = nn.Linear(32, 1)

#             def forward(self, x):
#                 h = self.net(x)
#                 mu = self.mu(h)
#                 logvar = self.logvar(h)
#                 std = torch.exp(0.5 * logvar)
#                 eps = torch.randn_like(std)
#                 z = mu + eps * std
#                 lambda_ = torch.sigmoid(z)
#                 return lambda_, mu, logvar

#         lambda_encoder = LambdaEncoder().to(device)
#         encoder_optimizer = torch.optim.Adam(lambda_encoder.parameters(), lr=1e-4)
#         lambda_prev = torch.tensor([0.5], device=device)

#         k = len(forget_dataset)
#         n = len(retain_dataset) + k
#         weighted_loss_retain_history = []
#         weighted_loss_forget_history = []
#         lambda_history = []

#         model.train()
#         epochs = 50

#         print("Starting SISS unlearning...")
#         for epoch in range(epochs):
#             progress_bar = tqdm(total=min(len(retain_loader), len(forget_loader)))
#             progress_bar.set_description(f"Epoch {epoch+1}")

#             retain_iter = iter(retain_loader)
#             forget_iter = iter(forget_loader)

#             for _ in range(min(len(retain_loader), len(forget_loader))):
#                 x_r, y_r = next(retain_iter)
#                 x_f, y_f = next(forget_iter)
#                 x_r, y_r, x_f, y_f = x_r.to(device), y_r.to(device), x_f.to(device), y_f.to(device)

#                 timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (x_r.size(0),), device=device).long()
#                 noise = torch.randn_like(x_r)
#                 noisy_from_retain = noise_scheduler.add_noise(x_r, noise, timesteps)
#                 noisy_from_forget = noise_scheduler.add_noise(x_f, noise, timesteps)
#                 mask = (torch.rand(x_r.size(0), 1, 1, 1, device=device) < lambda_prev.item()).float()
#                 m_t = (1 - mask) * noisy_from_retain + mask * noisy_from_forget
#                 y_m = ((1 - mask.squeeze()) * y_r.float() + mask.squeeze() * y_f.float()).long()

#                 emb_m = class_embedder(y_m).unsqueeze(1)
#                 noise_pred = model(m_t, timesteps, encoder_hidden_states=emb_m).sample

#                 alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
#                 sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[timesteps]).view(-1, 1, 1, 1)
#                 sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alphas_cumprod[timesteps]).view(-1, 1, 1, 1)
#                 variance_t = (sqrt_one_minus_alphas_cumprod_t**2).view(-1)
#                 dist_sq_x = torch.sum((m_t - sqrt_alphas_cumprod_t * x_r)**2, dim=(1,2,3))
#                 dist_sq_a = torch.sum((m_t - sqrt_alphas_cumprod_t * x_f)**2, dim=(1,2,3))
#                 log_q_x = -0.5 * dist_sq_x / variance_t
#                 log_q_a = -0.5 * dist_sq_a / variance_t
#                 log_denom = torch.logsumexp(torch.stack([
#                     torch.log(torch.tensor(1.0 - lambda_prev.item())) + log_q_x,
#                     torch.log(torch.tensor(lambda_prev.item())) + log_q_a]), dim=0)
#                 wx = torch.exp(log_q_x - log_denom)
#                 wa = torch.exp(log_q_a - log_denom)

#                 gt_noise_x = (m_t - sqrt_alphas_cumprod_t * x_r) / sqrt_one_minus_alphas_cumprod_t
#                 gt_noise_a = (m_t - sqrt_alphas_cumprod_t * x_f) / sqrt_one_minus_alphas_cumprod_t
#                 loss_x = nn.functional.mse_loss(noise_pred, gt_noise_x, reduction='none').mean(dim=(1,2,3))
#                 loss_a = nn.functional.mse_loss(noise_pred, gt_noise_a, reduction='none').mean(dim=(1,2,3))

#                 term_x = (n / (n - k)) * wx * loss_x
#                 term_a = ((1 + 5.0) * k / (n - k)) * wa * loss_a
#                 siss_loss = (term_x - term_a).mean()

#                 # Compute gradients before .backward()
#                 loss_x_mean = loss_x.mean()
#                 loss_a_mean = loss_a.mean()

#                 retain_grad = torch.autograd.grad(loss_x_mean, model.parameters(), retain_graph=True, allow_unused=True)
#                 forget_grad = torch.autograd.grad(loss_a_mean, model.parameters(), retain_graph=True, allow_unused=True)

#                 retain_grad_norm = torch.cat([g.flatten() for g in retain_grad if g is not None]).norm()
#                 forget_grad_norm = torch.cat([g.flatten() for g in forget_grad if g is not None]).norm()

#                 encoder_input = torch.tensor([
#                     loss_x_mean.item(),
#                     loss_a_mean.item(),
#                     retain_grad_norm.item(),
#                     forget_grad_norm.item()
#                 ], device=device).unsqueeze(0)

#                 lambda_new, mu, logvar = lambda_encoder(encoder_input)
#                 kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#                 encoder_loss = siss_loss.detach() + 0.01 * kl

#                 encoder_optimizer.zero_grad()
#                 encoder_loss.backward()
#                 encoder_optimizer.step()

#                 lambda_prev = lambda_new.detach().squeeze()
#                 lambda_history.append(lambda_prev.item())

#                 optimizer.zero_grad()
#                 siss_loss.backward()
#                 optimizer.step()

#                 weighted_loss_retain_history.append(term_x.mean().item())
#                 weighted_loss_forget_history.append(term_a.mean().item())

#                 progress_bar.set_postfix({
#                     "Retain Loss": weighted_loss_retain_history[-1],
#                     "Forget Loss": weighted_loss_forget_history[-1],
#                     "Lambda": lambda_prev.item()
#                 })
#                 progress_bar.update(1)

#             torch.cuda.empty_cache()

#         print("\nUnlearning finished.")

#         path = "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/DDPM_Unlearned_dynamic"
#         os.makedirs(path , exist_ok=True)

#         unet_path = "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/DDPM_Unlearned_dynamic/unet_unlearned_vae.pt"
#         cl_embedder_path = "/work/pi_aghasemi_umass_edu/afzali_umass/unlearning/new_task/conditional-diffusion-mnist/DDPM_Unlearned_dynamic/class_embedder_unlearned_vae.pt"
#         torch.save(model.state_dict(), unet_path)
#         torch.save(class_embedder.state_dict(), cl_embedder_path)
#         print("Unlearned model and class embedder saved.")

#         plt.figure(figsize=(12, 6))
#         plt.plot(weighted_loss_retain_history, label="Retain Loss Component")
#         plt.plot(weighted_loss_forget_history, label="Forget Loss Component")
#         plt.xlabel("Step")
#         plt.ylabel("Weighted MSE Loss")
#         plt.title("SISS Unlearning: Retain vs. Forget Loss Components")
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig("siss_loss_curves_dynamic.png")
#         print("Loss curves saved to siss_loss_curves.png")
#         plt.show()

#         plt.figure(figsize=(12, 6))
#         plt.plot(lambda_history, label="Lambda (Dynamic)", linestyle='--')
#         plt.xlabel("Step")
#         plt.ylabel("Lambda")
#         plt.title("SISS Unlearning: Lambda")
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig("siss_lambda.png")
#         print("Lambda saved to siss_lambda.png")
#         plt.show()
#     else:
#         raise NotImplementedError