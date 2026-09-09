# %%
import numpy as np 
import pandas as pd
import torch
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
from zcdp_accountant import compute_zcdp,get_privacy_spent
from Evaluation_metrics import compute_dimensionwise_probability ,compute_mmd
from Evaluation_metrics import compute_cluster_statistical_parity, silhouette_score, davies_bouldin_score
from Evaluation_metrics import compute_epsilon_identifiability, compute_nndr
from Evaluation_metrics import compute_beta_recall, compute_alpha_precision


# %%
if torch.cuda.is_available():
    print('Cuda is available')
    device = torch.device("cuda:0")
else:
    device = torch.device('cpu')
print(device)

# %%
dataset_directory = "C:/Users/Malek Adouani/Desktop/Evo_Clust_VAE_WGAN_GP/Dataset_Obesity/"
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

# %%
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
            nn.Linear(feature_dim, 512),  # Reduced number of neurons
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=self.dropout_prob),  # Add dropout to prevent overfitting
            nn.Linear(512, 256),  # Reduced number of neurons
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=self.dropout_prob),  # Add dropout to prevent overfitting
            nn.Linear(256, 2 * latent_dim)  # Output both mu and logvar
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),  # Reduced number of neurons
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=self.dropout_prob),  # Add dropout to prevent overfitting
            nn.Linear(256, 512),  # Reduced number of neurons
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=self.dropout_prob),  # Add dropout to prevent overfitting
            nn.Linear(512, feature_dim),
            nn.Sigmoid(),  # Use ReLU instead of Sigmoid
        )

        # Initialize cluster centroids
        self.register_buffer("cluster_centroids",torch.randn(num_clusters, latent_dim))


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
hyp_noise_multiplier = 0.01
latent_dim = 17
#n_epochs = 150
#Dataloaders
#TrainDataloader
dataset_train_object = Dataset(data=trainData, transform=False)
dataloader_train = DataLoader(dataset_train_object, batch_size=hyp_batch_size, shuffle=True, num_workers=0, drop_last=True)
#Dataset parameters
feature_s = dataset_train_object.featureSize
total_samples = len(dataset_train_object)
num_batches = len(dataloader_train)
iterations = total_samples * num_batches

# %%
import torch.nn as nn

def weights_init(m):
    """
    Custom weight initialization function.
    :param m: Module to initialize
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # Normal initialization for Conv layers
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)  # Use 0.0 bias for consistency
    elif classname.find('BatchNorm') != -1:
        # Constant initialization for BatchNorm layers
        nn.init.constant_(m.weight.data, 1.0)
        nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.Linear):
        # Xavier initialization for Linear layers
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)  # Use 0.0 bias for linear layers


# %%
CondVautoencoderModel = VAEWithClusters(feature_s ,latent_dim,num_clusters=20)
CondVautoencoderModel.apply(weights_init)
hyper = torch.FloatTensor
one = torch.FloatTensor([1])
mone = one * -1
CondVautoencoderModel.to(device)

# %%
#hyp_batch_size= 5
q = hyp_batch_size / total_samples
# Compute zCDP (rho)
rho = compute_zcdp(q, noise_multiplier=hyp_noise_multiplier, steps=10)
print(rho)
# Convert zCDP (rho) to (epsilon, delta) DP parameters
epsilon, delta, _ = get_privacy_spent(rho, target_delta=1e-5)
print(f"Achieves ({epsilon:.3f}, {delta:.1e})-DP")

# %%
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_optimizer(cls, epsilon_value, delta_value):
    class DPOptimizer(cls):
        def __init__(self, params, lr, betas, max_per_sample_grad_norm, noise_multiplier, batch_size, *args, **kwargs):
            super(DPOptimizer, self).__init__(params, lr=lr, betas=betas, *args, **kwargs)
            self.max_per_sample_grad_norm = max_per_sample_grad_norm
            self.noise_multiplier = noise_multiplier
            self.batch_size = batch_size
            self.epsilon_value = epsilon_value  # Store epsilon value
            self.delta_value = delta_value      # Store delta value

            for group in self.param_groups:
                group['aggregate_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]

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

                        # Compute the standard deviation for the Gaussian noise
                        std = self.max_per_sample_grad_norm * np.sqrt(2 * np.log(1.25 / self.delta_value)) / self.epsilon_value
                        
                        # Generate noise
                        noise = torch.normal(mean=0, std=std, size=param.grad.data.size(), device=device, dtype=param.grad.data.dtype)
                        
                        # Add noise to gradients
                        param.grad += noise / self.batch_size



        def step(self, *args, **kwargs):
            self.clip_grads_()
            self.add_noise_()
            super(DPOptimizer, self).step(*args, **kwargs)

    return DPOptimizer

AdamZCDP = create_optimizer(Adam,epsilon,delta)

optimizer_CVAE = AdamZCDP(
    CondVautoencoderModel.parameters(),
    lr=hyp_lr,
    betas=(hyp_b1, hyp_b2),
    max_per_sample_grad_norm=0.5,
    noise_multiplier=hyp_noise_multiplier,
    batch_size=hyp_batch_size,
   
)


# %%
def ConVAE_cluster_loss(x_recon, x_orig, mu, logvar, z, cluster_centroids, alpha=0.5,beta =0.01):
    # Reconstruction loss
    recon_loss = nn.functional.mse_loss(x_recon, x_orig, reduction='mean')

    # KL divergence loss
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Clustering loss
    if cluster_centroids is not None:
        # Compute pairwise distances (batch_size x num_clusters)
        distances = torch.cdist(z, cluster_centroids, p=2)  # Pairwise distances
        # Compute the minimum distance for each latent vector
        min_distances = torch.min(distances, dim=1).values
        # Cluster loss as the mean of minimum distances
        cluster_loss = torch.mean(min_distances)
    else:
        cluster_loss = 0.0

    

    return recon_loss +  alpha *kl_loss + beta * cluster_loss


# %%
def train_clust_vae(
    dataloader,
    feature_dim,
    latent_dim,
    num_clusters,
    n_epochs,
    device,
):
    model = VAEWithClusters(feature_dim, latent_dim, num_clusters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(n_epochs):
        for x in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar, z = model(x)
            loss = ConVAE_cluster_loss(
                x_recon, x, mu, logvar, z, model.cluster_centroids
            )
            loss.backward()
            optimizer.step()

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
# Define the column names of the new dataset
column_names = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
       'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
       'CALC', 'MTRANS', 'NObeyesdad']


train_protected_attributes = df[['Age', 'Gender']].to_numpy()


# %%
class DatasetWGAN:
    def __init__(self, data, protected_attributes, transform=None):
        # Transform
        self.transform = transform

        # load data here
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
# Assuming protected_attributes is a dictionary or tensor with both Ethnicity and Gender
# We will pass these attributes directly to DatasetWGAN
WGAN_GP_train_tensor = torch.Tensor(trainData)
dataset_train_object_wgan = DatasetWGAN(
    data=WGAN_GP_train_tensor,  # Input data
    protected_attributes=train_protected_attributes,  # Contains Ethnicity and Gender
    transform=False  # No additional transformation applied
)

dataloader_train_wgan = DataLoader(
    dataset_train_object_wgan, 
    batch_size=hyp_batch_size, 
    shuffle=True, 
    num_workers=0, 
    drop_last=True
)


# %%
import torch.nn as nn

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
def fairness_adversarial_loss(logits):
    """
    Generator-side fairness loss.
    Maximizes entropy of the fairness critic predictions.
    
    Parameters:
    - logits (torch.Tensor): Fairness critic logits (no labels).
    
    Returns:
    - torch.Tensor: Entropy-based adversarial loss.
    """
    probs = torch.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    return -entropy.mean()  # negative entropy → maximize entropy

def generator_loss(
    discriminator_predictions,
    fairness_logits,
    alpha=0.1
):
    """
    Generator loss for WGAN-GP with adversarial fairness.

    Parameters:
    - discriminator_predictions (torch.Tensor): Critic scores for fake samples.
    - fairness_logits (torch.Tensor): Fairness critic logits.
    - alpha (float): Fairness–utility trade-off weight.

    Returns:
    - torch.Tensor: Combined generator loss.
    """

    # WGAN adversarial loss
    adversarial_loss = -discriminator_predictions.mean()

    # Adversarial fairness loss (entropy maximization)
    fairness_loss = fairness_adversarial_loss(fairness_logits)

    # Combined loss
    return adversarial_loss + alpha * fairness_loss



def calc_gradient_penalty(netD, real_data, fake_data, device, lambda_gp=10):
    alpha = torch.rand(real_data.size(0), 1, device=device).expand_as(real_data)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.requires_grad_(True)
    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size(), device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty

def discriminator_loss_GP(netD, real_data, fake_data, real_predictions, fake_predictions, device, lambda_gp=10):
    """
    Compute the discriminator loss with gradient penalty.

    Parameters:
    - netD (torch.nn.Module): Discriminator model.
    - real_data (torch.Tensor): Real data samples.
    - fake_data (torch.Tensor): Fake data samples generated by the generator.
    - real_predictions (torch.Tensor): Discriminator's predictions for real data.
    - fake_predictions (torch.Tensor): Discriminator's predictions for fake data.
    - device (torch.device): Computation device (CPU/GPU).
    - lambda_gp (float): Weight for the gradient penalty.

    Returns:
    - total_loss (torch.Tensor): Total discriminator loss.
    """
    # Wasserstein loss
    d_loss = fake_predictions.mean() - real_predictions.mean()

    # Gradient penalty
    gradient_penalty = calc_gradient_penalty(netD, real_data, fake_data, device, lambda_gp)

    # Combine losses
    total_loss = d_loss + gradient_penalty
    return total_loss



# %%
def pad_clusters(cluster_one_hot, max_clusters):
    B, K = cluster_one_hot.shape
    if K < max_clusters:
        pad = torch.zeros(B, max_clusters - K, device=cluster_one_hot.device)
        return torch.cat([cluster_one_hot, pad], dim=1)
    return cluster_one_hot


# %%
def compute_cluster_assignments(x, K):
    """
    x: torch.Tensor [B, d]
    K: int (EA decision variable)
    """
    with torch.no_grad():
        kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
        labels = kmeans.fit_predict(x.detach().cpu().numpy())

    labels = torch.tensor(labels, device=x.device)
    return F.one_hot(labels, num_classes=K).float()


# %%
MAX_CLUSTERS = 20  # upper bound searched by EA
N_SENSITIVE_ATTRS = 2  # e.g. Gender, Ethnicity
hidden_dim = dataset_train_object_wgan.return_protected_attributes().shape[1]
wgan_input_size = dataset_train_object_wgan.featureSize

generatorModel = Generator(wgan_input_size).to(device)
discriminatorModel = Discriminator(wgan_input_size).to(device)

fairnessCriticModel = FairnessCritic(
    max_clusters=MAX_CLUSTERS,
    n_sensitive_attrs=N_SENSITIVE_ATTRS
).to(device)


# %%
b1 = 0.5
b2 = 0.999
sample_interval = 100
weight_decay = 0.0001
hyp_lr = 1e-4
beta = 0.5  # Weight for feature matching loss
optimizer_G = torch.optim.Adam(generatorModel.parameters(), lr=1e-4, betas=(b1, b2), weight_decay=weight_decay)
optimizer_FC = torch.optim.Adam(fairnessCriticModel.parameters(), lr=hyp_lr, betas=(b1, b2),weight_decay=weight_decay)
optimizer_D = torch.optim.Adam(discriminatorModel.parameters(), lr=hyp_lr, betas=(b1, b2), weight_decay=weight_decay)

# %%
generatorModel.apply(weights_init)
discriminatorModel.apply(weights_init)
fairnessCriticModel.apply(weights_init)

# %%
Tensor = torch.FloatTensor
one = torch.FloatTensor([1])
mone = one * -1

# %%

def compute_cluster_labels(data, num_clusters, max_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(data.detach().cpu().numpy())

    labels = torch.tensor(labels, dtype=torch.long, device=data.device)
    one_hot = F.one_hot(labels, num_classes=num_clusters).float()

    # critical fix
    one_hot = pad_clusters(one_hot, max_clusters)

    return one_hot


# %%
def train_wgan_fair(
    generator,
    discriminator,
    fairness_critic,
    dataloader,
    optimizer_G,
    optimizer_D,
    optimizer_FC,
    device,
    num_epochs=20,
    num_clusters=10,
    lambda_gp=10.0,
    alpha_fair=0.1,
    l1_weight=1e-3,
    sample_interval=64
):
    generator.train()
    discriminator.train()
    fairness_critic.train()

    for epoch in range(num_epochs):
        for i_batch, (real_data, protected_attributes) in enumerate(dataloader):

            real_data = real_data.to(device)

            # ---------------------
            # 1. Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            noise = torch.randn(real_data.size(0), real_data.size(1), device=device)
            fake_data = generator(noise).detach()

            real_scores = discriminator(real_data)
            fake_scores = discriminator(fake_data)

            d_loss = discriminator_loss_GP(
                discriminator,
                real_data,
                fake_data,
                real_scores,
                fake_scores,
                device,
                lambda_gp=lambda_gp
            )

            d_loss.backward()
            optimizer_D.step()

            # ---------------------
            # 2. Train Fairness Critic (supervised, real data)
            # ---------------------
            optimizer_FC.zero_grad()

            with torch.no_grad():
                cluster_labels_real = compute_cluster_labels(
                    real_data,
                    num_clusters=num_clusters,
                    max_clusters=MAX_CLUSTERS
                )


            fairness_logits_real = fairness_critic(cluster_labels_real)

            protected_dict = {
                
                "Gender": protected_attributes[:, 0].to(device),
                "Age": protected_attributes[:, 1].to(device),
            }

            fc_loss = fairness_adversarial_loss(
                fairness_logits_real,
                
            )

            l1_fc = sum(p.abs().sum() for p in fairness_critic.parameters())
            (fc_loss + l1_weight * l1_fc).backward()
            optimizer_FC.step()

            # ---------------------
            # 3. Train Generator
            # ---------------------
            optimizer_G.zero_grad()

            noise = torch.randn(real_data.size(0), real_data.size(1), device=device)
            fake_data = generator(noise)

            fake_scores = discriminator(fake_data)

            cluster_labels_fake = compute_cluster_labels(
                        fake_data,
                        num_clusters=num_clusters,
                        max_clusters=MAX_CLUSTERS
                    )


            fairness_logits_fake = fairness_critic(cluster_labels_fake)

            g_loss = generator_loss(
                fake_scores,
                fairness_logits_fake,
                alpha=alpha_fair
            )

            l1_g = sum(p.abs().sum() for p in generator.parameters())
            (g_loss + l1_weight * l1_g).backward()
            optimizer_G.step()

            # ---------------------
            # Logging
            # ---------------------
            batches_done = epoch * len(dataloader) + i_batch
            if batches_done % sample_interval == 0:
                wasserstein = real_scores.mean().item() - fake_scores.mean().item()
                print(
                    f"[Epoch {epoch+1}/{num_epochs}] "
                    f"[Batch {i_batch}] "
                    f"[D: {d_loss.item():.3f}] "
                    f"[G: {g_loss.item():.3f}] "
                    f"[FC: {fc_loss.item():.3f}] "
                    f"[W-dist: {wasserstein:.3f}]",
                    flush=True
                )


# %%
def train_wgan_fair_wrapper(
    reconstructed_data,
    protected_attributes,
    num_clusters,
    lambda_gp,
    alpha_fair,
    n_epochs,
    device,
):
    dataset = DatasetWGAN(
        torch.tensor(reconstructed_data, dtype=torch.float32),
        protected_attributes
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

    G = Generator(dataset.featureSize).to(device)
    D = Discriminator(dataset.featureSize).to(device)
    FC = FairnessCritic(MAX_CLUSTERS, N_SENSITIVE_ATTRS).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=1e-4)
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-4)
    opt_FC = torch.optim.Adam(FC.parameters(), lr=1e-4)

    train_wgan_fair(
        G, D, FC,
        dataloader,
        opt_G, opt_D, opt_FC,
        device,
        num_epochs=n_epochs,
        num_clusters=num_clusters,
        lambda_gp=lambda_gp,
        alpha_fair=alpha_fair
    )

    return G


# %%
def save_models(generator, discriminator, fairness_critic, path, tag):
    torch.save(generator.state_dict(), f"{path}/generator_{tag}.pth")
    torch.save(discriminator.state_dict(), f"{path}/discriminator_{tag}.pth")
    torch.save(fairness_critic.state_dict(), f"{path}/fairness_critic_{tag}.pth")

def load_models(generator, discriminator, fairness_critic, path, tag, device):
    generator.load_state_dict(torch.load(f"{path}/generator_{tag}.pth"))
    discriminator.load_state_dict(torch.load(f"{path}/discriminator_{tag}.pth"))
    fairness_critic.load_state_dict(torch.load(f"{path}/fairness_critic_{tag}.pth"))

    generator.to(device).eval()
    discriminator.to(device).eval()
    fairness_critic.to(device).eval()


# %%
def fairness_objective(X, cluster_labels, sensitive_attrs=None):
    """
    Post-hoc fairness proxy for UNSUPERVISED synthetic data.
    No protected attributes are assumed available.
    """

    from sklearn.metrics import silhouette_score, davies_bouldin_score

    ss = silhouette_score(X, cluster_labels)
    dbi = davies_bouldin_score(X, cluster_labels)

    # Lower is better (consistent with minimization)
    return (1.0 - ss) + dbi


def privacy_objective(real, synthetic):
    eps_risk = compute_epsilon_identifiability(real, synthetic)
    nndr = compute_nndr(real, synthetic)
    return eps_risk + float(np.mean(nndr))

def utility_objective(real, synthetic):
    mmd = compute_mmd(real, synthetic)
    dwp = compute_dimensionwise_probability(real, synthetic)
    alpha_p = compute_alpha_precision(real, synthetic, alpha=0.5)
    beta_r  = compute_beta_recall(real, synthetic, beta=0.5)

    return (
        mmd +
        float(np.mean(dwp)) -
        alpha_p -
        beta_r
    )



# %%
def sample_generator(G, n_samples, device):
    G.eval()
    z = torch.randn(n_samples, feature_s, device=device)
    with torch.no_grad():
        x_gen = G(z)
    return x_gen.cpu().numpy()


# %%
def get_cluster_labels(clust_vae, X, device):
    clust_vae.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        # Forward pass returns reconstructed X, latent variables z, maybe other stuff
        outputs = clust_vae(X_tensor)
        # Extract latent embeddings (check your VAE implementation)
        z = outputs[1]  # e.g., if outputs = (recon, z, mu, logvar)
        cluster_labels = clust_vae.assign_clusters(z)
    return cluster_labels.cpu().numpy()


# %%
from pymoo.core.problem import Problem
import torch
import numpy as np
from torch.utils.data import DataLoader
import logging

# -------------------------------------------------
# Logging (minimal, paper-ready)
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)
logger = logging.getLogger("NSGA2-Clust_VAE_WGAN_GP")


class GenerativeTradeoffProblem(Problem):

    def __init__(self):
        super().__init__(
            n_var=5,
            n_obj=3,
            xl=[5, 1.0, 0.01, 5, 5],
            xu=[20, 20.0, 1.0, 30, 30]
        )

    def _evaluate(self, X, out, *args, **kwargs):

        F = []

        for i, (K, lambda_gp, alpha_fair, n_C, n_G) in enumerate(X):

            logger.info(
                f"[Eval {i}] K={int(K)}, λ_gp={lambda_gp:.2f}, "
                f"α_fair={alpha_fair:.3f}, n_C={int(n_C)}, n_G={int(n_G)}"
            )

            # -------------------------------------------------
            # 1. Train ClustVAE on real data
            # -------------------------------------------------
            clust_vae = train_clust_vae(
                dataloader=dataloader_train,
                feature_dim=feature_s,
                latent_dim=latent_dim,
                num_clusters=int(K),
                n_epochs=int(n_C),
                device=device
            )

            # -------------------------------------------------
            # 2. Infer clusters on real data (for WGAN training)
            # -------------------------------------------------
            recon_real, _, cluster_labels_real = infer_with_clusters(
                clust_vae,
                dataloader_train,
                device
            )

            # -------------------------------------------------
            # 3. Train fairness-aware WGAN (fairness enforced HERE)
            # -------------------------------------------------
            G = train_wgan_fair_wrapper(
                reconstructed_data=recon_real,
                protected_attributes=train_protected_attributes,
                num_clusters=int(K),
                lambda_gp=lambda_gp,
                alpha_fair=alpha_fair,
                n_epochs=int(n_G),
                device=device
            )

            # -------------------------------------------------
            # 4. Generate synthetic data
            # -------------------------------------------------
            X_real = (
                trainData.cpu().numpy()
                if torch.is_tensor(trainData)
                else trainData
            )

            X_syn = sample_generator(G, len(X_real), device)

            # -------------------------------------------------
            # 5. Infer cluster labels for synthetic data
            # -------------------------------------------------
            dataloader_syn = DataLoader(
                torch.tensor(X_syn, dtype=torch.float32),
                batch_size=64
            )

            _, _, cluster_labels_syn = infer_with_clusters(
                clust_vae,
                dataloader_syn,
                device
            )

            cluster_labels_syn = np.asarray(cluster_labels_syn).ravel()

            # -------------------------------------------------
            # 6. Objectives (NO synthetic sensitive attributes)
            # -------------------------------------------------
            f_util = utility_objective(X_real, X_syn)

            # Fairness proxy = clustering structure only
            # (fairness already enforced adversarially during training)
            f_fair = fairness_objective(
                X_syn,
                cluster_labels_syn,
                sensitive_attrs=None  # <- explicitly unused
            )

            f_priv = privacy_objective(X_real, X_syn)

            logger.info(
                f"[Result {i}] Utility={f_util:.4f} | "
                f"Fairness={f_fair:.4f} | Privacy={f_priv:.4f}"
            )

            F.append([f_util, f_fair, f_priv])

        out["F"] = np.asarray(F)


# %%
import random
import time
import numpy as np
import pandas as pd
import torch

# IMPORTANT:
# Restore F as torch.nn.functional.
# Existing functions such as compute_cluster_labels()
# currently call F.one_hot(...).
import torch.nn.functional as F

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize


# ============================================================
# Safety check for torch.nn.functional
# ============================================================
# This immediately catches any accidental reassignment of F
# before starting the expensive 30-run experiment.
assert hasattr(F, "one_hot"), (
    "ERROR: F is no longer torch.nn.functional. "
    "Do not use the variable name F for Pareto fronts."
)


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Experimental configuration
# ============================================================

N_RUNS = 30

# Reduced NSGA-II budget
POP_SIZE = 4
N_EVALS = 12

# Approximately:
# 4 individuals × 3 generations = 12 evaluations/run
#
# Instead of:
# 8 individuals × 10 generations ≈ 80 evaluations/run


# ============================================================
# Storage
# ============================================================

all_fronts = []
all_results = []
run_summary = []


# ============================================================
# 30 independent NSGA-II runs
# ============================================================

total_start = time.time()


for run_seed in range(N_RUNS):

    run_start = time.time()

    print("\n" + "=" * 70)
    print(
        f"NSGA-II RUN {run_seed + 1}/{N_RUNS} | "
        f"seed={run_seed}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Defensive check
    # --------------------------------------------------------
    # Existing functions in the notebook use F.one_hot().
    # Stop immediately if F has accidentally been overwritten.
    if not hasattr(F, "one_hot"):
        raise RuntimeError(
            "F has been overwritten and is no longer "
            "torch.nn.functional."
        )

    # --------------------------------------------------------
    # Seed stochastic pipeline
    # --------------------------------------------------------
    set_seed(run_seed)

    # --------------------------------------------------------
    # Fresh NSGA-II instance
    # --------------------------------------------------------
    algorithm = NSGA2(
        pop_size=POP_SIZE,
        eliminate_duplicates=True
    )

    # --------------------------------------------------------
    # Fresh problem instance
    # --------------------------------------------------------
    problem = GenerativeTradeoffProblem()

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------
    res = minimize(
        problem,
        algorithm,
        termination=("n_eval", N_EVALS),
        seed=run_seed,
        verbose=False
    )

    # --------------------------------------------------------
    # Pareto front
    # --------------------------------------------------------
    # IMPORTANT:
    # Never write:
    #
    #     F = np.asarray(res.F)
    #
    # because F must remain torch.nn.functional.
    # --------------------------------------------------------

    pareto_front = np.asarray(res.F)

    all_results.append(res)
    all_fronts.append(pareto_front)

    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------
    run_seconds = time.time() - run_start
    run_minutes = run_seconds / 60.0

    n_evaluations = res.algorithm.evaluator.n_eval

    # --------------------------------------------------------
    # Store run information
    # --------------------------------------------------------
    run_summary.append({
        "run": run_seed + 1,
        "seed": run_seed,
        "n_evaluations": n_evaluations,
        "n_pareto_solutions": len(pareto_front),
        "runtime_minutes": run_minutes
    })

    # --------------------------------------------------------
    # Display run summary
    # --------------------------------------------------------
    print(
        f"\nRun {run_seed + 1}/{N_RUNS} completed"
        f"\nEvaluations          : {n_evaluations}"
        f"\nPareto solutions     : {len(pareto_front)}"
        f"\nRuntime              : {run_minutes:.2f} min"
    )

    # --------------------------------------------------------
    # SAVE AFTER EVERY RUN
    # --------------------------------------------------------
    # Saving after every independent run prevents losing all
    # completed results if a later run crashes.
    # --------------------------------------------------------

    np.save(
        f"NSGAII_front_seed_{run_seed}.npy",
        pareto_front
    )

    pd.DataFrame(run_summary).to_csv(
        "NSGAII_30runs_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # GPU cleanup between runs
    # --------------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Final summary
# ============================================================

total_hours = (time.time() - total_start) / 3600.0

summary_df = pd.DataFrame(run_summary)


print("\n" + "=" * 70)
print("NSGA-II 30-RUN EXPERIMENT COMPLETED")
print("=" * 70)

print(summary_df)


print(
    f"\nMean runtime/run : "
    f"{summary_df['runtime_minutes'].mean():.2f} min"
)

print(
    f"Std runtime/run  : "
    f"{summary_df['runtime_minutes'].std(ddof=1):.2f} min"
)

print(
    f"Total runtime    : "
    f"{total_hours:.2f} hours"
)

# %%
from pymoo.indicators.hv import HV

# -------------------------------------------------
# Pool all Pareto fronts
# -------------------------------------------------
F_all = np.vstack(all_fronts)

# Common normalization bounds
ideal = F_all.min(axis=0)
nadir = F_all.max(axis=0)

denominator = nadir - ideal
denominator[denominator == 0] = 1.0


# Reference point in normalized objective space
ref_point = np.array([1.1, 1.1, 1.1])

hv_indicator = HV(ref_point=ref_point)

hv_values = []


for run_id, F in enumerate(all_fronts):

    # Normalize using COMMON bounds
    F_normalized = (F - ideal) / denominator

    hv = hv_indicator(F_normalized)

    hv_values.append(hv)

    print(
        f"Run {run_id + 1:02d}: "
        f"HV = {hv:.6f}"
    )


hv_values = np.array(hv_values)

print("\nNSGA-II Hypervolume over 30 runs")
print("--------------------------------")
print(f"Mean HV : {hv_values.mean():.6f}")
print(f"Std HV  : {hv_values.std(ddof=1):.6f}")
print(f"Min HV  : {hv_values.min():.6f}")
print(f"Max HV  : {hv_values.max():.6f}")
