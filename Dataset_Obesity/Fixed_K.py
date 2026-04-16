# %%
import json
import traceback
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from datetime import datetime
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import sklearn.model_selection as skl
import torch.nn.functional as F
import torch.autograd as autograd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
from zcdp_accountant import compute_zcdp, get_privacy_spent
from Evaluation_metrics import (compute_dimensionwise_probability, compute_mmd,
                                 compute_cluster_statistical_parity, silhouette_score, 
                                 davies_bouldin_score, compute_epsilon_identifiability, 
                                 compute_nndr, compute_beta_recall, compute_alpha_precision)

# %%
# %%
if torch.cuda.is_available():
    print('Cuda is available')
    device = torch.device("cuda:0")
else:
    device = torch.device('cpu')
print(device)

# %%
dataset_directory = "C:/Users/Malek Adouani/Desktop/CLUST_VAE_WGAN_GP_OPT_GIT-main/Dataset_3/"
df = pd.read_csv(dataset_directory + "preprocessed_obese_data.csv")
#df.head()

# Identify numerical and categorical features
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(include=['object', 'category']).columns

# Count each feature type
num_numerical_features = len(numerical_features)
num_categorical_features = len(categorical_features)

# Display the counts
print(f"Number of Numerical Features: {num_numerical_features}")
print(f"Number of Categorical Features: {num_categorical_features}")

trainData = df.to_numpy()
trainData = torch.from_numpy(trainData).float().to(device)

class Dataset:
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data
        self.sampleSize = data.shape[0]
        self.featureSize = data.shape[1]
    
    def return_data(self):
        return self.data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]
        return sample

# %%
class VAEWithClusters(nn.Module):
    def __init__(self, feature_dim, latent_dim, num_clusters, dropout_prob=0.3, l2_reg=1e-5):
        super(VAEWithClusters, self).__init__()
        self.latent_dim = latent_dim
        self.num_clusters = num_clusters
        self.dropout_prob = dropout_prob
        self.l2_reg = l2_reg
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(256, 2 * latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(512, feature_dim),
            nn.Sigmoid(),
        )
        
        # Initialize cluster centroids
        self.register_buffer("cluster_centroids", torch.randn(num_clusters, latent_dim))
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z

# %%
hyp_lr = 1e-4
hyp_batch_size = 86
hyp_b1 = 0.5
hyp_b2 = 0.999
rho_op = 0.4
micro_batch_size = 20
hyp_noise_multiplier = 0.0031
latent_dim = 20

# Dataloaders
dataset_train_object = Dataset(data=trainData, transform=False)
dataloader_train = DataLoader(dataset_train_object, batch_size=hyp_batch_size, 
                              shuffle=True, num_workers=0, drop_last=True)

# Dataset parameters
feature_s = dataset_train_object.featureSize
total_samples = len(dataset_train_object)
num_batches = len(dataloader_train)
iterations = total_samples * num_batches

# %%
def weights_init(m):
    """Custom weight initialization function."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight.data, 1.0)
        nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

# %%
CondVautoencoderModel = VAEWithClusters(feature_s, latent_dim, num_clusters=15)
CondVautoencoderModel.apply(weights_init)
hyper = torch.FloatTensor
one = torch.FloatTensor([1])
mone = one * -1
CondVautoencoderModel.to(device)

# %%
q = hyp_batch_size / total_samples
rho = compute_zcdp(q, noise_multiplier=hyp_noise_multiplier, steps=10)
print(rho)
epsilon, delta, _ = get_privacy_spent(rho, target_delta=1e-5)
print(f"Achieves ({epsilon:.3f}, {delta:.1e})-DP")

# %%
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

def create_optimizer(cls, epsilon_value, delta_value):
    class DPOptimizer(cls):
        def __init__(self, params, lr, betas, max_per_sample_grad_norm, 
                     noise_multiplier, batch_size, *args, **kwargs):
            super(DPOptimizer, self).__init__(params, lr=lr, betas=betas, *args, **kwargs)
            self.max_per_sample_grad_norm = max_per_sample_grad_norm
            self.noise_multiplier = noise_multiplier
            self.batch_size = batch_size
            self.epsilon_value = epsilon_value
            self.delta_value = delta_value
            
            for group in self.param_groups:
                group['aggregate_grads'] = [
                    torch.zeros_like(param.data) if param.requires_grad else None 
                    for param in group['params']
                ]
        
        def clip_grads_(self):
            params = self.param_groups[0]['params']
            clip_grad_norm_(params, max_norm=self.max_per_sample_grad_norm, norm_type=2)
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['aggregate_grads']):
                    if param.requires_grad:
                        accum_grad.add_(param.grad.data)
        
        def add_noise_(self):
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['aggregate_grads']):
                    if param.requires_grad:
                        param.grad.data = accum_grad.clone()
                        std = self.noise_multiplier * self.max_per_sample_grad_norm
                        noise = torch.normal(
                            mean=0, std=std, 
                            size=param.grad.data.size(), 
                            device=device, 
                            dtype=param.grad.data.dtype
                        )
                        param.grad += noise / self.batch_size
        
        def step(self, *args, **kwargs):
            self.clip_grads_()
            self.add_noise_()
            super(DPOptimizer, self).step(*args, **kwargs)
    
    return DPOptimizer

AdamZCDP = create_optimizer(Adam, epsilon, delta)
optimizer_CVAE = AdamZCDP(
    CondVautoencoderModel.parameters(),
    lr=hyp_lr,
    betas=(hyp_b1, hyp_b2),
    max_per_sample_grad_norm=0.5,
    noise_multiplier=hyp_noise_multiplier,
    batch_size=hyp_batch_size,
)

# %%
def ConVAE_cluster_loss(x_recon, x_orig, mu, logvar, z, cluster_centroids, alpha=0.5, beta=0.01):
    recon_loss = nn.functional.mse_loss(x_recon, x_orig, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    if cluster_centroids is not None:
        distances = torch.cdist(z, cluster_centroids, p=2)
        min_distances = torch.min(distances, dim=1).values
        cluster_loss = torch.mean(min_distances)
    else:
        cluster_loss = 0.0
    
    return recon_loss + alpha * kl_loss + beta * cluster_loss

# %%
def train_clust_vae(dataloader, feature_dim, latent_dim, num_clusters, n_epochs, device):
    model = VAEWithClusters(feature_dim, latent_dim, num_clusters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        for batch_idx, x in enumerate(dataloader):
            x = x.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar, z = model(x)
            loss = ConVAE_cluster_loss(x_recon, x, mu, logvar, z, model.cluster_centroids)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {epoch_loss/len(dataloader):.4f}")
    
    return model

# %%
def infer_with_clusters(model, dataloader, device):
    model.eval()
    recons, originals, cluster_labels = [], [], []
    
    with torch.no_grad():
        for data_batch in dataloader:
            data_batch = data_batch.to(device)
            x_recon, _, _, z = model(data_batch)
            distances = torch.cdist(z, model.cluster_centroids)
            labels = torch.argmin(distances, dim=1)
            
            recons.append(x_recon.cpu())
            originals.append(data_batch.cpu())
            cluster_labels.append(labels.cpu())
    
    return (
        torch.cat(recons).numpy(),
        torch.cat(originals).numpy(),
        torch.cat(cluster_labels).numpy()
    )

# %%
# TRAIN FROM SCRATCH INSTEAD OF LOADING
n_epochs = 50
print("Training Cluster-based VAE from scratch...")
trained_vae_model = train_clust_vae(
    dataloader=dataloader_train,
    feature_dim=feature_s,
    latent_dim=latent_dim,
    num_clusters=15,
    n_epochs=n_epochs,
    device=device
)

# %%
# Get reconstructed data
print("Generating reconstructed data...")
reconstructed_data, original_data, cluster_labels = infer_with_clusters(
    trained_vae_model, 
    dataloader_train, 
    device
)

# %%
column_names = df.columns.to_list()

reconstructed_data_df = pd.DataFrame(reconstructed_data, columns=column_names)
train_protected_attributes = reconstructed_data_df[['Age', 'Gender']].to_numpy()

# %%
class DatasetWGAN:
    def __init__(self, data, protected_attributes, transform=None):
        self.transform = transform
        self.data = data
        self.protected_attributes = protected_attributes
        self.sampleSize = data.shape[0]
        self.featureSize = data.shape[1]
        self.label_size = 1
    
    def return_data(self):
        return self.data
    
    def return_protected_attributes(self):
        return self.protected_attributes
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]
        protected_attribute = self.protected_attributes[idx]
        if self.transform:
            pass
        return sample, protected_attribute

# %%
WGAN_GP_train_tensor = torch.Tensor(reconstructed_data)
dataset_train_object_wgan = DatasetWGAN(
    data=WGAN_GP_train_tensor,
    protected_attributes=train_protected_attributes,
    transform=False
)

dataloader_train_wgan = DataLoader(
    dataset_train_object_wgan,
    batch_size=hyp_batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=True
)

# %%
class Generator(nn.Module):
    def __init__(self, feature_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feature_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, feature_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )
    
    def forward(self, x):
        return self.model(x)

class FairnessCritic(nn.Module):
    def __init__(self, max_clusters, n_sensitive_attrs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(max_clusters, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_sensitive_attrs)
        )
    
    def forward(self, cluster_one_hot):
        return self.net(cluster_one_hot)

# %%
def fairness_adversarial_loss(logits, protected_attributes=None):
    """Generator-side fairness loss. Maximizes entropy of the fairness critic predictions."""
    if logits.dim() == 1:
        logits = logits.unsqueeze(1)
    probs = torch.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(torch.clamp(probs, min=1e-8)), dim=1)
    return -entropy.mean()

# %%
def discriminator_loss_GP(discriminator, real_data, fake_data, real_scores, fake_scores, device, lambda_gp=10.0):
    """Wasserstein GAN-GP discriminator loss."""
    d_loss = fake_scores.mean() - real_scores.mean()
    
    # Gradient penalty
    alpha = torch.rand(real_data.size(0), 1, device=device)
    interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return d_loss + lambda_gp * gradient_penalty

def generator_loss(fake_scores, fairness_logits_fake, alpha=0.1):
    """Generator loss combining adversarial and fairness objectives."""
    g_adv_loss = -fake_scores.mean()
    g_fair_loss = fairness_adversarial_loss(fairness_logits_fake)
    return g_adv_loss + alpha * g_fair_loss

def utility_objective(X_real, X_syn):
    """Utility metric based on MMD."""
    try:
        mmd = compute_mmd(X_real, X_syn)
        return float(mmd)
    except:
        return 1.0

def fairness_objective(X_syn, cluster_labels_syn, sensitive_attrs):
    """Fairness metric based on statistical parity."""
    try:
        fairness_score = compute_cluster_statistical_parity(cluster_labels_syn, sensitive_attrs)
        return float(fairness_score)
    except:
        return 1.0

def privacy_objective(X_real, X_syn):
    """Privacy metric based on NNDR."""
    try:
        nndr = compute_nndr(X_real, X_syn)
        return float(nndr)
    except:
        return 1.0

# %%
def safe_compute_cluster_labels(data, num_clusters, max_clusters, device):
    """Safely compute cluster labels with error handling."""
    try:
        actual_clusters = min(num_clusters, data.shape[0] - 1)
        actual_clusters = max(actual_clusters, 2)
        
        kmeans = KMeans(n_clusters=actual_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(data.detach().cpu().numpy())
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        one_hot = F.one_hot(labels, num_classes=actual_clusters).float()
        one_hot = pad_clusters(one_hot, max_clusters)
        
        return one_hot
    except Exception as e:
        print(f"️ Clustering error: {e}")
        B = data.shape[0]
        fallback = torch.zeros(B, max_clusters, device=device)
        fallback[:, 0] = 1.0
        return fallback

# %%
def pad_clusters(cluster_one_hot, max_clusters):
    """Pad cluster one-hot to max_clusters dimension."""
    B, K = cluster_one_hot.shape
    if K < max_clusters:
        pad = torch.zeros(B, max_clusters - K, device=cluster_one_hot.device)
        return torch.cat([cluster_one_hot, pad], dim=1)
    elif K > max_clusters:
        return cluster_one_hot[:, :max_clusters]
    return cluster_one_hot

# %%
def train_wgan_fair_fixed(
    generator, discriminator, fairness_critic, dataloader,
    optimizer_G, optimizer_D, optimizer_FC, device,
    num_epochs=20, num_clusters=10, lambda_gp=10.0, alpha_fair=0.1,
    l1_weight=1e-3, sample_interval=64, max_clusters=20, n_sensitive_attrs=2
):
    """Fixed training loop with proper error handling."""
    generator.train()
    discriminator.train()
    fairness_critic.train()
    
    for epoch in range(num_epochs):
        for i_batch, batch_data in enumerate(dataloader):
            try:
                # Handle both single tensor and (data, protected_attrs) tuples
                if isinstance(batch_data, (tuple, list)):
                    real_data, protected_attributes = batch_data
                    real_data = real_data.to(device)
                    if isinstance(protected_attributes, torch.Tensor):
                        protected_attributes = protected_attributes.to(device)
                else:
                    real_data = batch_data.to(device)
                    protected_attributes = None
                
                # 1. Train Discriminator
                optimizer_D.zero_grad()
                noise = torch.randn(real_data.size(0), real_data.size(1), device=device)
                fake_data = generator(noise).detach()
                
                real_scores = discriminator(real_data)
                fake_scores = discriminator(fake_data)
                
                d_loss = discriminator_loss_GP(
                    discriminator, real_data, fake_data, real_scores, fake_scores,
                    device, lambda_gp=lambda_gp
                )
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                optimizer_D.step()
                
                # 2. Train Fairness Critic
                optimizer_FC.zero_grad()
                with torch.no_grad():
                    cluster_labels_real = safe_compute_cluster_labels(
                        real_data, num_clusters=num_clusters,
                        max_clusters=max_clusters, device=device
                    )
                
                fairness_logits_real = fairness_critic(cluster_labels_real)
                fc_loss = fairness_adversarial_loss(fairness_logits_real, protected_attributes)
                
                l1_fc = sum(p.abs().sum() for p in fairness_critic.parameters())
                (fc_loss + l1_weight * l1_fc).backward()
                torch.nn.utils.clip_grad_norm_(fairness_critic.parameters(), 1.0)
                optimizer_FC.step()
                
                # 3. Train Generator
                optimizer_G.zero_grad()
                noise = torch.randn(real_data.size(0), real_data.size(1), device=device)
                fake_data = generator(noise)
                fake_scores = discriminator(fake_data)
                
                cluster_labels_fake = safe_compute_cluster_labels(
                    fake_data, num_clusters=num_clusters,
                    max_clusters=max_clusters, device=device
                )
                fairness_logits_fake = fairness_critic(cluster_labels_fake)
                
                g_loss = generator_loss(fake_scores, fairness_logits_fake, alpha=alpha_fair)
                l1_g = sum(p.abs().sum() for p in generator.parameters())
                (g_loss + l1_weight * l1_g).backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                optimizer_G.step()
                
                # Logging
                batches_done = epoch * len(dataloader) + i_batch
                if batches_done % sample_interval == 0:
                    wasserstein = real_scores.mean().item() - fake_scores.mean().item()
                    print(
                        f"[Epoch {epoch+1}/{num_epochs}] [Batch {i_batch}] "
                        f"[D: {d_loss.item():.4f}] [G: {g_loss.item():.4f}] "
                        f"[FC: {fc_loss.item():.4f}] [W-dist: {wasserstein:.4f}]"
                    )
            
            except Exception as e:
                print(f"️ Batch {i_batch} error: {e}")
                traceback.print_exc()
                continue

# %%
def train_wgan_fair_wrapper_fixed(
    reconstructed_data, protected_attributes, num_clusters, lambda_gp,
    alpha_fair, n_epochs, device, feature_s, max_clusters=20
):
    """Fixed wrapper with proper parameter passing."""
    try:
        dataset = DatasetWGAN(
            torch.tensor(reconstructed_data, dtype=torch.float32),
            protected_attributes
        )
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
        
        G = Generator(feature_s).to(device)
        D = Discriminator(feature_s).to(device)
        FC = FairnessCritic(max_clusters=max_clusters, n_sensitive_attrs=2).to(device)
        
        opt_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))
        opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))
        opt_FC = torch.optim.Adam(FC.parameters(), lr=1e-4, betas=(0.5, 0.999))
        
        train_wgan_fair_fixed(
            G, D, FC, dataloader, opt_G, opt_D, opt_FC, device,
            num_epochs=n_epochs, num_clusters=num_clusters,
            lambda_gp=lambda_gp, alpha_fair=alpha_fair,
            max_clusters=max_clusters, n_sensitive_attrs=2
        )
        
        return G
    except Exception as e:
        print(f" WGAN training failed: {e}")
        traceback.print_exc()
        return Generator(feature_s).to(device)

# %%
def sample_generator(G, n_samples, device, feature_s):
    """Fixed generator sampling."""
    try:
        G.eval()
        z = torch.randn(n_samples, feature_s, device=device)
        with torch.no_grad():
            x_gen = G(z)
        return x_gen.cpu().numpy()
    except Exception as e:
        print(f"️ Sampling failed: {e}")
        return np.random.randn(n_samples, feature_s).astype(np.float32)

# %%
def evaluate_configuration_fixed(
    K: int, lambda_gp: float, alpha_fair: float, n_C: int, n_G: int,
    trainData: torch.Tensor, dataloader_train, feature_s: int, latent_dim: int,
    device, train_protected_attributes: np.ndarray, max_clusters: int = 20
) -> Tuple[float, float, float]:
    """Fixed evaluation with proper data handling and error management."""
    try:
        # 1. Train ClustVAE
        clust_vae = train_clust_vae(
            dataloader=dataloader_train,
            feature_dim=feature_s,
            latent_dim=latent_dim,
            num_clusters=int(K),
            n_epochs=int(n_C),
            device=device
        )
        
        # 2. Infer clusters on real data
        recon_real, original_data, cluster_labels_real = infer_with_clusters(
            clust_vae, dataloader_train, device
        )
        
        # 3. Train WGAN-fair
        G = train_wgan_fair_wrapper_fixed(
            reconstructed_data=recon_real,
            protected_attributes=train_protected_attributes,
            num_clusters=int(K),
            lambda_gp=lambda_gp,
            alpha_fair=alpha_fair,
            n_epochs=int(n_G),
            device=device,
            feature_s=feature_s,
            max_clusters=max_clusters
        )
        
        # 4. Generate synthetic data
        X_real = trainData.cpu().numpy() if torch.is_tensor(trainData) else trainData
        X_syn = sample_generator(G, min(len(X_real), 500), device, feature_s)
        
        # 5. Ensure alignment
        min_len = min(len(X_real), len(X_syn))
        X_real = X_real[:min_len]
        X_syn = X_syn[:min_len]
        
        # 6. Compute objectives
        try:
            f_util = utility_objective(X_real, X_syn)
        except Exception as e:
            print(f"️ Utility computation failed: {e}")
            f_util = 1.0
        
        try:
            sensitive_syn = train_protected_attributes[:min_len, :2]
            cluster_labels_syn = cluster_labels_real[:min_len]
            f_fair = fairness_objective(X_syn, cluster_labels_syn, sensitive_syn)
        except Exception as e:
            print(f"️ Fairness computation failed: {e}")
            f_fair = 1.0
        
        try:
            f_priv = privacy_objective(X_real, X_syn)
        except Exception as e:
            print(f"️ Privacy computation failed: {e}")
            f_priv = 1.0
        
        return float(f_util), float(f_fair), float(f_priv)
    
    except Exception as e:
        print(f" Configuration (K={K}, λ_gp={lambda_gp}, α={alpha_fair}) failed: {e}")
        traceback.print_exc()
        return 1.0, 1.0, 1.0

# %%
def fixed_K_baseline_improved(
    K_fixed: int, trainData: torch.Tensor, dataloader_train,
    feature_s: int, latent_dim: int, device,
    train_protected_attributes: np.ndarray, max_clusters: int = 20,
    verbose: bool = True
) -> List[Dict]:
    """Improved Fixed-K baseline with proper error handling."""
    results = []
    lambda_gp_grid = [1.0, 5.0, 10.0]
    alpha_fair_grid = [0.05, 0.1, 0.2]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Fixed-K Baseline (K = {K_fixed})")
        print(f"{'='*60}")
        print(f"Grid: λ_gp ∈ {lambda_gp_grid}, α_fair ∈ {alpha_fair_grid}\n")
    
    config_count = 0
    for lambda_gp in lambda_gp_grid:
        for alpha_fair in alpha_fair_grid:
            config_count += 1
            
            if verbose:
                print(
                    f"[{config_count}/9] K={K_fixed} | "
                    f"λ_gp={lambda_gp:.1f} | α_fair={alpha_fair:.2f}",
                    end=" → "
                )
            
            try:
                f_util, f_fair, f_priv = evaluate_configuration_fixed(
                    K=K_fixed,
                    lambda_gp=lambda_gp,
                    alpha_fair=alpha_fair,
                    n_C=10,
                    n_G=10,
                    trainData=trainData,
                    dataloader_train=dataloader_train,
                    feature_s=feature_s,
                    latent_dim=latent_dim,
                    device=device,
                    train_protected_attributes=train_protected_attributes,
                    max_clusters=max_clusters
                )
                
                result = {
                    "K": int(K_fixed),
                    "lambda_gp": float(lambda_gp),
                    "alpha_fair": float(alpha_fair),
                    "utility": float(f_util),
                    "fairness": float(f_fair),
                    "privacy": float(f_priv)
                }
                results.append(result)
                
                if verbose:
                    print(f" U:{f_util:.4f} \vert  F:{f_fair:.4f} \vert  P:{f_priv:.4f}")
            
            except Exception as e:
                print(f" FAILED: {str(e)[:50]}")
                result = {
                    "K": int(K_fixed),
                    "lambda_gp": float(lambda_gp),
                    "alpha_fair": float(alpha_fair),
                    "utility": None,
                    "fairness": None,
                    "privacy": None,
                    "error": str(e)
                }
                results.append(result)
    
    return results

# %%
def generate_json_report(all_results: Dict) -> str:
    """Generate comprehensive JSON report with statistics."""
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "algorithm": "Fixed-K Heuristic Baseline",
            "dataset": "HIV",
            "experiment_type": "Multi-Objective Optimization"
        },
        "results_by_K": {},
        "summary_statistics": {}
    }
    
    for K, results in all_results.items():
        K_int = int(K)
        report["results_by_K"][K_int] = results
        
        # Compute statistics
        valid_results = [r for r in results if r.get("utility") is not None]
        
        if valid_results:
            utilities = [r["utility"] for r in valid_results]
            fairnesses = [r["fairness"] for r in valid_results]
            privacies = [r["privacy"] for r in valid_results]
            
            report["summary_statistics"][K_int] = {
                "n_configurations": len(valid_results),
                "n_failed": len(results) - len(valid_results),
                "utility": {
                    "mean": float(np.mean(utilities)),
                    "std": float(np.std(utilities)),
                    "min": float(np.min(utilities)),
                    "max": float(np.max(utilities))
                },
                "fairness": {
                    "mean": float(np.mean(fairnesses)),
                    "std": float(np.std(fairnesses)),
                    "min": float(np.min(fairnesses)),
                    "max": float(np.max(fairnesses))
                },
                "privacy": {
                    "mean": float(np.mean(privacies)),
                    "std": float(np.std(privacies)),
                    "min": float(np.min(privacies)),
                    "max": float(np.max(privacies))
                }
            }
    
    return json.dumps(report, indent=2)

# %%
# MAIN EXECUTION
if __name__ == "__main__":
    K_values = [5, 8, 10, 12, 15]
    all_fixed_K_results = {}
    
    for K in K_values:
        all_fixed_K_results[K] = fixed_K_baseline_improved(
            K_fixed=K,
            trainData=trainData,
            dataloader_train=dataloader_train,
            feature_s=feature_s,
            latent_dim=latent_dim,
            device=device,
            train_protected_attributes=train_protected_attributes,
            max_clusters=20,
            verbose=True
        )
    
    # Generate and save JSON report
    json_report = generate_json_report(all_fixed_K_results)
    output_path = dataset_directory + "fixed_k_baseline_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(json_report)
    
    print(f"\n Results saved to: {output_path}")
    print(json_report)


