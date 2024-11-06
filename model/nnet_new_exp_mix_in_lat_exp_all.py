import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.cluster import KMeans
import scipy
from lightning import LightningModule

# Function to apply Gumbel softmax to multiple sets of logits and select top N indices
def muti_gumbel(logits, tau=1, hard=False, eps=1e-10, dim=-1, top_N=10, num_use_moe=10):
    """
    Applies Gumbel softmax to multiple sets of logits and selects top N indices for each.
    Returns hard and soft masks.

    Args:
        logits (Tensor): Input logits of shape (batch_size, num_use_moe, num_features).
        tau (float): Temperature parameter for Gumbel softmax.
        hard (bool): Whether to return hard one-hot samples.
        eps (float): Small value to avoid numerical issues (deprecated).
        dim (int): Dimension along which softmax is applied.
        top_N (int): Number of top indices to select.
        num_use_moe (int): Number of mixtures of experts.

    Returns:
        mask (Tensor): Hard masks of shape (batch_size, num_use_moe, num_features).
        mask_soft (Tensor): Soft masks of shape (batch_size, num_use_moe, num_features).
    """
    mask_list = []
    mask_soft_list = []
    for i in range(num_use_moe):
        # Apply Gumbel softmax to each set of logits
        mask_soft, mask = gumbel_softmax_topN(logits[:, i, :], tau=tau, hard=hard, eps=eps, dim=dim, top_N=top_N)
        mask_list.append(mask)
        mask_soft_list.append(mask_soft)
    # Stack masks along new dimension
    return torch.stack(mask_list, dim=1), torch.stack(mask_soft_list, dim=1)

# Function to perform Gumbel softmax sampling and select top N indices
def gumbel_softmax_topN(logits, tau=1, hard=False, eps=1e-10, dim=-1, top_N=10):
    """
    Performs Gumbel softmax sampling and selects top N indices.

    Args:
        logits (Tensor): Input logits of shape (batch_size, num_features).
        tau (float): Temperature parameter.
        hard (bool): Whether to return hard one-hot samples.
        eps (float): Small value to avoid numerical issues (deprecated).
        dim (int): Dimension along which softmax is applied.
        top_N (int): Number of top indices to select.

    Returns:
        y_soft (Tensor): Softmax probabilities after Gumbel noise is added.
        ret (Tensor): Hard or soft samples depending on 'hard' flag.
    """
    # Note: 'eps' parameter is deprecated and has no effect
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    # Sample Gumbel noise
    gumbels = -torch.empty_like(logits).exponential_().log()
    # Add Gumbel noise to logits and scale by temperature
    gumbels = (logits + gumbels) / tau
    # Apply softmax
    y_soft = gumbels.softmax(dim)

    if hard:
        # Get top N indices
        index = y_soft.topk(k=top_N, dim=dim)[1]
        # Create hard one-hot encoding
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        # Straight-through estimator
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Return soft probabilities
        ret = y_soft
    return y_soft, ret

# Cosine annealing learning rate scheduler with warmup
class CosineAnnealingSchedule(_LRScheduler):
    """Cosine annealing with warmup."""

    def __init__(
        self, opt, final_lr=0, n_epochs=1000, warmup_epochs=10, warmup_lr=0
    ):
        """
        Initializes the scheduler.

        Args:
            opt (Optimizer): Optimizer.
            final_lr (float): Final learning rate after decay.
            n_epochs (int): Total number of epochs.
            warmup_epochs (int): Number of warmup epochs.
            warmup_lr (float): Initial learning rate for warmup.
        """
        self.opt = opt
        self.optimizer = self.opt
        self.base_lr = base_lr = opt.defaults["lr"]
        self.final_lr = final_lr
        self.n_epochs = n_epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr

        # Compute number of decay epochs
        decay_epochs = 1 + n_epochs - warmup_epochs
        self.decay_epochs = decay_epochs

        # Warmup schedule: linearly increase lr from warmup_lr to base_lr
        warmup_schedule = np.linspace(warmup_lr, base_lr, warmup_epochs)
        # Decay schedule: cosine annealing from base_lr to final_lr
        decay_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
            1 + np.cos(np.pi * np.arange(decay_epochs) / decay_epochs)
        )
        # Concatenate warmup and decay schedules
        self.lr_schedule = np.hstack((warmup_schedule, decay_schedule))

        self._last_lr = self.lr_schedule[0]
        self.cur_epoch = 0

        self.init_opt()

    def init_opt(self):
        """Initializes the optimizer learning rate."""
        self.step()
        # self.set_epoch(0)

    def get_lr(self):
        """Gets the current learning rate."""
        return self.lr_schedule[self.cur_epoch]

    def step(self):
        """Updates the learning rate for the optimizer."""
        lr = self.get_lr()
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

        self.cur_epoch += 1
        self._last_lr = lr
        return lr

    def set_epoch(self, epoch):
        """Sets the current epoch (for resuming training)."""
        self.cur_epoch = epoch

# Define a neural network module with Linear, BatchNorm, and LeakyReLU layers
class NN_FCBNRL_MM(nn.Module):
    """
    Neural network module consisting of Linear, BatchNorm, Dropout, and LeakyReLU layers.
    """

    def __init__(self, in_dim, out_dim, channel=8, use_RL=True, use_BN=True, use_DO=True):
        """
        Initializes the module.

        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.
            channel (int): Unused parameter.
            use_RL (bool): Whether to use LeakyReLU activation.
            use_BN (bool): Whether to use BatchNorm1d.
            use_DO (bool): Whether to use Dropout.
        """
        super(NN_FCBNRL_MM, self).__init__()
        layers = []
        # Linear layer
        layers.append(nn.Linear(in_dim, out_dim))
        # Optional Dropout
        if use_DO:
            layers.append(nn.Dropout(p=0.02))
        # Optional BatchNorm
        if use_BN:
            layers.append(nn.BatchNorm1d(out_dim))
        # Optional LeakyReLU activation
        if use_RL:
            layers.append(nn.LeakyReLU(0.1))
        
        # Create the sequential block
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass of the module."""
        return self.block(x)

# Transformer Encoder with optional Mixture of Experts (MoE)
class TransformerEncoder(nn.Module):
    """
    Transformer Encoder module with optional Mixture of Experts (MoE).
    """

    def __init__(
        self, 
        num_layers=2, 
        num_attention_heads=6, 
        hidden_size=240, 
        intermediate_size=300, 
        max_position_embeddings=784, 
        num_input_dim=784,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        num_use_moe=10,
        use_moe=True,
    ):
        """
        Initializes the Transformer Encoder.

        Args:
            num_layers (int): Number of layers.
            num_attention_heads (int): Number of attention heads.
            hidden_size (int): Hidden size.
            intermediate_size (int): Intermediate size.
            max_position_embeddings (int): Maximum position embeddings.
            num_input_dim (int): Input dimension size.
            hidden_dropout_prob (float): Dropout probability for hidden layers.
            attention_probs_dropout_prob (float): Dropout probability for attention.
            num_use_moe (int): Number of experts in MoE.
            use_moe (bool): Whether to use Mixture of Experts.
        """
        super(TransformerEncoder, self).__init__()
        self.use_moe = use_moe
        
        # Determine the type of network to use based on input dimension
        if num_input_dim == 3072:
            nn_type = 'resnet'
            print('Using ResNet')
        else:
            nn_type = 'nn'
            print('Using fully connected network')
            
        if self.use_moe:
            # Create a list of encoders for MoE
            self.enc = torch.nn.ModuleList([
                self.network_single(
                    num_input_dim,
                    hidden_size,
                    num_layers,
                    nn_type=nn_type,
                ) for _ in range(num_use_moe)
            ])
        else:
            # Single encoder
            self.enc = self.network_single(
                num_input_dim, 
                hidden_size, 
                num_layers,
                nn_type=nn_type,
            )
        
        # Output fully connected layer
        self.fc = nn.Sequential(
            NN_FCBNRL_MM(hidden_size, num_input_dim, use_RL=False),
        )

    def network_single(self, num_input_dim, hidden_size, num_layers, nn_type='nn'):
        """
        Creates a single network (either ResNet or fully connected).

        Args:
            num_input_dim (int): Input dimension.
            hidden_size (int): Hidden size.
            num_layers (int): Number of layers.
            nn_type (str): Type of network ('nn' or 'resnet').

        Returns:
            enc (nn.Module): The network module.
        """
        if nn_type == 'resnet':
            # Use ResNet architecture
            enc = ResNet(BasicBlock, [2, 2, 2, 2], 3)
        else:
            # Build fully connected network
            layers = []
            layers.append(NN_FCBNRL_MM(num_input_dim, hidden_size))
            for _ in range(num_layers):
                layers.append(
                    NN_FCBNRL_MM(hidden_size, hidden_size)
                )
            layers.append(NN_FCBNRL_MM(hidden_size, hidden_size, use_RL=False))
            enc = nn.Sequential(*layers)
        return enc
    
    def forward(self, input_x):
        """
        Forward pass of the Transformer Encoder.

        Args:
            input_x (Tensor): Input tensor of shape (batch_size, num_use_moe, ...).

        Returns:
            emb (Tensor): Output embeddings.
        """
        if self.use_moe:
            # If using MoE, apply each expert to the input
            emb_all = [self.fc(enc(input_x[:, i, :])) for i, enc in enumerate(self.enc)]
            emb = torch.stack(emb_all, dim=1)
        else:
            # Single encoder
            emb = self.fc(self.enc(input_x))
        return emb

# Main model class
class DMTEVT_model(LightningModule):
    """
    DMTEVT_model is a PyTorch Lightning module that implements the training and evaluation of the model.
    """

    def __init__(
        self,
        lr=0.005,
        sigma=0.05,
        sample_rate_feature=0.6,
        num_input_dim=64,
        num_train_data=60000,
        weight_decay=0.0001,
        exaggeration_lat=1,
        exaggeration_emb=1,
        weight_mse=2,
        weight_nepo=1,
        nu_lat=0.1,
        nu_emb=0.1,
        tau=1,
        T_num_layers=2,
        T_num_attention_heads=6,
        T_hidden_size=240,
        T_intermediate_size=300,
        T_hidden_dropout_prob=0.1,
        T_attention_probs_dropout_prob=0.1,
        ckpt_path=None,
        use_orthogonal=False,
        num_use_moe=1,
        vis_dim=2,
        trans_out_dim=50,
        max_epochs=600,
        v_latent=0.01,
        n_neg_sample=4,
        test_noise=False,
        **kwargs,
    ):
        """
        Initializes the model with given hyperparameters.

        Args:
            lr (float): Learning rate.
            sigma (float): Sigma parameter for similarity function.
            sample_rate_feature (float): Sampling rate for features.
            num_input_dim (int): Input dimension size.
            num_train_data (int): Number of training data samples.
            weight_decay (float): Weight decay for optimizer.
            exaggeration_lat (float): Exaggeration parameter for latent space.
            exaggeration_emb (float): Exaggeration parameter for embedding space.
            weight_mse (float): Weight for MSE loss.
            weight_nepo (float): Weight for NEPO loss.
            nu_lat (float): Degrees of freedom for t-distribution in latent space.
            nu_emb (float): Degrees of freedom for t-distribution in embedding space.
            tau (float): Temperature parameter.
            T_num_layers (int): Number of layers in Transformer.
            T_num_attention_heads (int): Number of attention heads in Transformer.
            T_hidden_size (int): Hidden size in Transformer.
            T_intermediate_size (int): Intermediate size in Transformer.
            T_hidden_dropout_prob (float): Dropout probability in Transformer.
            T_attention_probs_dropout_prob (float): Dropout probability for attention in Transformer.
            ckpt_path (str): Path to checkpoint for loading model.
            use_orthogonal (bool): Whether to use orthogonal loss.
            num_use_moe (int): Number of experts in Mixture of Experts.
            vis_dim (int): Dimension of visualization space.
            trans_out_dim (int): Output dimension of Transformer.
            max_epochs (int): Maximum number of epochs.
            v_latent (float): Variance parameter in latent space.
            n_neg_sample (int): Number of negative samples.
            test_noise (bool): Whether to test with noise.
            **kwargs: Additional arguments.
        """
        super().__init__()

        self.setup_bool_zzl = False
        self.save_hyperparameters()

        num_input_dim = self.hparams.num_input_dim
        self.init_exp_bool = False
        
        # Initialize the encoder
        self.enc = TransformerEncoder(
            num_layers=T_num_layers, 
            num_attention_heads=T_num_attention_heads, 
            hidden_size=T_hidden_size,
            intermediate_size=T_intermediate_size,
            max_position_embeddings=20,
            num_input_dim=num_input_dim,
            hidden_dropout_prob=T_hidden_dropout_prob,
            attention_probs_dropout_prob=T_attention_probs_dropout_prob,
            num_use_moe=num_use_moe,
        )
        
        # Visualization network
        self.vis = self.InitNetworkMLP(NS=[num_input_dim*num_use_moe, 500, vis_dim], last_relu=False)
        # Embedding layer for experts
        self.exp = nn.Embedding(self.hparams.num_use_moe, num_input_dim)

        # Load checkpoint if provided
        if ckpt_path is not None:
            print('Loading checkpoint from:', ckpt_path)
            self.load_state_dict(torch.load(ckpt_path))

    def InitNetworkMLP(self, NS, last_relu=True, use_DO=True, use_BN=True, use_RL=True):
        """
        Initializes a multi-layer perceptron (MLP) network.

        Args:
            NS (list): List of layer sizes.
            last_relu (bool): Whether to use ReLU activation on the last layer.
            use_DO (bool): Whether to use Dropout.
            use_BN (bool): Whether to use BatchNorm.
            use_RL (bool): Whether to use LeakyReLU activation.

        Returns:
            model_pat (nn.Sequential): The MLP network.
        """
        layers = []
        for i in range(len(NS) - 1):
            # Determine if last layer should have activation
            if i == len(NS) - 2 and not last_relu:
                layers.append(NN_FCBNRL_MM(NS[i], NS[i + 1], use_RL=False, use_DO=use_DO, use_BN=use_BN))
            else:
                layers.append(NN_FCBNRL_MM(NS[i], NS[i + 1], use_RL=use_RL, use_DO=use_DO, use_BN=use_BN))
        model_pat = nn.Sequential(*layers)
        return model_pat

    def _DistanceSquared(self, x, y=None, metric="euclidean"):
        """
        Computes squared Euclidean distance between samples.

        Args:
            x (Tensor): Input tensor of shape (n_samples, n_features).
            y (Tensor): Optional second input tensor.
            metric (str): Distance metric to use ('euclidean').

        Returns:
            dist (Tensor): Distance matrix.
        """
        if metric == "euclidean":
            if y is not None:
                m, n = x.size(0), y.size(0)
                xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
                yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
                dist = xx + yy
                dist = torch.addmm(dist, mat1=x, mat2=y.t(), beta=1, alpha=-2)
                dist = dist.clamp(min=1e-12)
            else:
                m, n = x.size(0), x.size(0)
                xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
                yy = xx.t()
                dist = xx + yy
                dist = torch.addmm(dist, mat1=x, mat2=x.t(), beta=1, alpha=-2)
                dist = dist.clamp(min=1e-12)
                dist[torch.eye(dist.shape[0], dtype=torch.bool)] = 1e-12
        return dist

    def _CalGamma(self, v):
        """
        Calculates the gamma function value.

        Args:
            v (float): Degrees of freedom.

        Returns:
            out (float): Gamma function value.
        """
        a = scipy.special.gamma((v + 1) / 2)
        b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
        out = a / b
        return out

    def _Similarity(self, dist, sigma=0.3):
        """
        Computes similarity using Gaussian kernel.

        Args:
            dist (Tensor): Distance matrix.
            sigma (float): Standard deviation of the Gaussian kernel.

        Returns:
            Pij (Tensor): Similarity matrix.
        """
        dist = dist.clamp(min=0)
        Pij = torch.exp(-dist / (2 * sigma ** 2))
        return Pij

    def t_distribution_similarity(self, distance_matrix, df):
        """
        Computes similarity matrix using t-distribution kernel.

        Args:
            distance_matrix (Tensor): Distance matrix.
            df (float): Degrees of freedom for t-distribution.

        Returns:
            similarity_matrix (Tensor): Similarity matrix.
        """
        distance_matrix = distance_matrix + 1e-6
        numerator = (1 + distance_matrix ** 2 / df) ** (-(df + 1) / 2)
        denominator = torch.sum(numerator, dim=1, keepdim=True) - torch.diagonal(numerator, 0).unsqueeze(1)
        similarity_matrix = numerator / denominator
        return similarity_matrix

    def LossManifold(self, latent_data, temperature=1, exaggeration=1, nu=0.1):
        """
        Computes the manifold loss between two views of the data.

        Args:
            latent_data (Tensor): Latent representations of shape (2 * batch_size, ...).
            temperature (float): Temperature scaling.
            exaggeration (float): Exaggeration factor.
            nu (float): Degrees of freedom for t-distribution.

        Returns:
            loss (Tensor): Computed loss.
        """
        batch_size = latent_data.shape[0] // 2        
        features_a = latent_data[:batch_size]
        features_b = latent_data[batch_size:]

        # Compute pairwise distances
        dis_aa = torch.cdist(features_a, features_a) * temperature
        dis_bb = torch.cdist(features_b, features_b) * temperature
        dis_ab = torch.cdist(features_a, features_b) * temperature

        # Compute similarity matrices using t-distribution
        sim_aa = self.t_distribution_similarity(dis_aa, df=nu)
        sim_bb = self.t_distribution_similarity(dis_bb, df=nu)
        sim_ab = self.t_distribution_similarity(dis_ab, df=nu)

        # Compute alignment term
        tempered_alignment = (torch.diagonal(sim_ab).log()).mean()

        # Exclude self similarities
        self_mask = torch.eye(batch_size, dtype=bool, device=sim_aa.device)
        sim_aa.masked_fill_(self_mask, 0.0)
        sim_bb.masked_fill_(self_mask, 0.0)

        # Compute uniformity terms
        logsumexp_1 = torch.hstack((sim_ab.T, sim_bb)).sum(1).log_().mean()
        logsumexp_2 = torch.hstack((sim_aa, sim_ab)).sum(1).log_().mean()

        raw_uniformity = logsumexp_1 + logsumexp_2

        # Compute final loss
        loss = -(exaggeration * tempered_alignment - raw_uniformity / 2)

        return loss

    def batch_patten_loss(self, feature_tra, mask):
        """
        Computes orthogonal loss to encourage diversity among experts.

        Args:
            feature_tra (Tensor): Transformed features.
            mask (Tensor): Masks indicating selected features.

        Returns:
            loss (Tensor): Computed loss.
        """
        # Add small noise to features
        feature_tra = feature_tra + torch.randn_like(feature_tra) * 0.001 * feature_tra.std()
        batch_size = feature_tra.shape[0] // 8
        feature_tra = feature_tra[:batch_size]
        mask = mask[:batch_size]

        mean_value_list = []
        for i in range(feature_tra.shape[1]):
            fea_ins = feature_tra[:, i, :]
            mask_ins = mask[:, i, :] == 1
            fea_ins_umask = fea_ins[mask_ins == 1].reshape((feature_tra.shape[0], -1))
            # Compute cosine similarity
            cosine_similarity_matrix = torch.nn.functional.cosine_similarity(
                fea_ins_umask.unsqueeze(1),
                fea_ins_umask.unsqueeze(0),
                dim=2
            )
            upper_triangular_matrix_no_diag = torch.triu(cosine_similarity_matrix, diagonal=1)
            mean_value = upper_triangular_matrix_no_diag.mean()
            mean_value_list.append(mean_value)
        
        # Return the mean of the mean values
        return 1 + torch.stack(mean_value_list).mean()
    
    def forward(self, x, tau=100.0):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input data.
            tau (float): Temperature parameter for Gumbel softmax.

        Returns:
            x_masked (Tensor): Masked input data.
            lat_higt_dim_out (Tensor): High-dimensional latent outputs.
            lat_vis (Tensor): Low-dimensional visualization outputs.
            lat_high_dim (Tensor): High-dimensional latent representations.
        """
        batch_size = x.shape[0] // 2

        if len(x.shape) == 2:
            # Determine number of features to select
            num_select = int(x.shape[1] * self.hparams.sample_rate_feature)
            # Get weights for Gumbel softmax
            weight = self.get_weight()
            # Apply Gumbel softmax to get masks
            self.mask, self.soft_mask = muti_gumbel(
                weight,
                tau=tau,
                hard=True,
                top_N=num_select,
                num_use_moe=self.hparams.num_use_moe,
            )
            # Expand masks to match batch size
            self.mask = self.mask.expand(x.shape[0], -1, -1)
            if self.mask.shape[0] != x.shape[0]:
                self.mask = torch.cat([self.mask, self.mask])
            
            if len(x.shape) == 2:
                # Apply masks to input data
                x_masked = torch.einsum('bik,bk->bik', self.mask, x)
            elif len(x.shape) == 4:
                # Handle image data
                _, num_channel, num_x, num_y = x.shape
                x = x.reshape((x.shape[0], -1))
                x_masked = torch.einsum('bik,bk->bik', self.mask, x)
                x_masked = x_masked.reshape((batch_size * 2, self.hparams.num_use_moe, num_channel, num_x, num_y))
        else:
            # For other data shapes
            x_masked = x.unsqueeze(1).expand(-1, self.hparams.num_use_moe, -1, -1, -1)
            weight = x_masked

        # Pass through encoder
        lat_higt_dim_out = self.enc(x_masked)
        # Reshape outputs
        lat_high_dim = lat_higt_dim_out.reshape((batch_size * 2, -1))
        # Pass through visualization network
        lat_vis = self.vis(lat_high_dim)
        return x_masked, lat_higt_dim_out, lat_vis, lat_high_dim

    def get_weight(self):
        """
        Retrieves and processes the expert weights.

        Returns:
            weight (Tensor): Processed weights.
        """
        w = self.exp(torch.arange(self.hparams.num_use_moe).to(self.device)).reshape(1, self.hparams.num_use_moe, -1)
        weight = F.tanh(w) * 10
        return weight

    def get_tau(self, epoch, total_epochs=900, tau_start=100, tau_end=1.001):
        """
        Computes the temperature parameter tau for Gumbel softmax.

        Args:
            epoch (int): Current epoch.
            total_epochs (int): Total number of epochs.
            tau_start (float): Initial tau value.
            tau_end (float): Final tau value.

        Returns:
            tau (float): Computed tau value.
        """
        if epoch >= total_epochs:
            return tau_end
        else:    
            return tau_start * (tau_end / tau_start) ** (epoch / (total_epochs - 1))

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.

        Args:
            batch (dict): Batch of data.
            batch_idx (int): Batch index.

        Returns:
            loss_all (Tensor): Computed loss.
        """
        data_input_item = batch['data_input_item']
        data_input_aug = batch['data_input_aug']
        index = batch['index']
        
        # Initialize expert embeddings with KMeans clustering
        if not self.init_exp_bool:
            data_np = data_input_item.cpu().numpy().T
            kmeans = KMeans(n_clusters=self.hparams.num_use_moe, random_state=0)
            kmeans.fit(data_np)

            labels = kmeans.labels_
            cluster_matrix = np.zeros((data_np.shape[0], kmeans.n_clusters))
            for idx, label in enumerate(labels):
                cluster_matrix[idx, label] = 1

            cluster_matrix_tensor = torch.tensor(cluster_matrix).float()
            self.exp.weight.data = cluster_matrix_tensor.t().to(self.device)
            self.init_exp_bool = True

        # Concatenate original and augmented data
        data_input_item = torch.cat([data_input_item, data_input_aug])
        # Forward pass
        x_masked, lat_high_dim_exp, lat_vis, _ = self(
            data_input_item, 
            tau=self.hparams.tau,
        )
        # Compute mean over experts
        lat_high_dim = lat_high_dim_exp.mean(dim=1)
        
        # Compute orthogonal loss if required
        if self.hparams.use_orthogonal:
            orthogonal_loss = self.batch_patten_loss(x_masked, self.mask)
        else:
            orthogonal_loss = 0
        
        # Compute manifold losses
        loss_lat = self.LossManifold(
            latent_data=lat_high_dim.reshape(lat_high_dim.shape[0], -1),
            temperature=1,
            exaggeration=self.hparams.exaggeration_lat,
            nu=self.hparams.nu_lat,
        )
        loss_emb = self.LossManifold(
            latent_data=lat_vis.reshape(lat_vis.shape[0], -1),
            temperature=1,
            exaggeration=self.hparams.exaggeration_emb,
            nu=self.hparams.nu_emb,
        )
        
        # Compute total loss
        loss_all = (loss_emb + loss_lat) / 2 + orthogonal_loss * 10

        # Log losses
        self.log('loss_all', loss_all, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('loss_emb', loss_emb, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('loss_lat', loss_lat, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('orthogonal_loss', orthogonal_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('lr', float(self.trainer.optimizers[0].param_groups[0]["lr"]), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss_all

    def validation_step(self, batch, batch_idx, test=False, dataloader_idx=0):
        """
        Performs a validation step.

        Args:
            batch (dict): Batch of data.
            batch_idx (int): Batch index.
            test (bool): Whether this is a test step.
            dataloader_idx (int): Index of the dataloader.

        Returns:
            None
        """
        if dataloader_idx == 0:
            data_input_item = batch['data_input_item']
            data_input_aug = batch['data_input_aug']
            index = batch['index']

            x_masked, lat_high_dim_exp, lat_vis, lat_high_dim = self(
                data_input_item,
                tau=self.hparams.tau,
            )
            
            if self.hparams.test_noise:
                noist_test_result_dict = []
                for i in range(5):
                    noist_test_result = self.noise_map(lat_high_dim, noise_level=i*0.1+0.1)
                    noist_test_result_dict.append(noist_test_result)
                self.noist_test_result_dict = torch.stack(noist_test_result_dict).cpu()
            # Store outputs for further processing
            self.validation_origin_input = data_input_item
            self.validation_step_outputs_high = lat_high_dim
            self.validation_step_outputs_vis = lat_vis
            self.validation_step_lat_vis_exp = lat_vis
            self.validation_weight = self.get_weight()[0]
    
    def test_step(self, batch, batch_idx):
        """
        Performs a test step.

        Args:
            batch (dict): Batch of data.
            batch_idx (int): Batch index.

        Returns:
            None
        """
        data_input_item = batch['data_input_item']
        data_input_aug = batch['data_input_aug']
        label = batch['label']
        
        x_masked, lat_high_dim, lat_vis, _ = self(
            data_input_item,
        )

        # Store outputs for further processing
        self.test_step_outputs_high = lat_high_dim
        self.test_step_outputs_vis = lat_vis
        self.test_step_outputs_label = label

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            dict: Dictionary containing optimizer and scheduler.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            weight_decay=self.hparams.weight_decay, 
            lr=self.hparams.lr
        )
        lrsched = CosineAnnealingSchedule(
            optimizer, n_epochs=self.hparams.max_epochs, warmup_epochs=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lrsched,
                "interval": "epoch",
            },  # interval "step" for batch update
        }

    def noise_map(self, data, num_exp=10, noise_level=0.1):
        """
        Tests the robustness of the embeddings to noise.

        Args:
            data (Tensor): Input data.
            num_exp (int): Number of experiments.
            noise_level (float): Level of noise to add.

        Returns:
            distance_tensor (Tensor): Tensor containing distances.
        """
        exp_feature_num = int(data.shape[1] // num_exp)

        emb = self.vis(data)
        
        distance_list = []
        for i in range(num_exp):
            start_index = i * exp_feature_num
            end_index = (i + 1) * exp_feature_num
            noise_data_delta = torch.rand_like(data) * noise_level * data.std(dim=0)
            noise_data = torch.clone(data)
            noise_data[:, start_index:end_index] += noise_data_delta[:, start_index:end_index]
            noise_emb = self.vis(noise_data)
            distance = torch.norm(noise_emb - emb, dim=1)
            distance_list.append(distance)
        
        distance_tensor = torch.stack(distance_list, dim=1)
        return distance_tensor