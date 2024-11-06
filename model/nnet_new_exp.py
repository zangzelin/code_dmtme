import numpy as np
from lightning import LightningModule
import os
import sys
from PIL import Image
import wandb
import platform
import torch.nn.functional as F
import plotly.graph_objects as go
import lightning as pl
import torch
from torch import nn
import torch.distributed as distributed
from lightning.pytorch.cli import LightningCLI
import torchvision.utils as vutils
from torchvision import models
from lightning.pytorch.strategies.single_device import SingleDeviceStrategy

from lightning import Trainer

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from plotly.subplots import make_subplots
import functools
from call_backs.MaskExp import MaskExpCallBack
from transformers import BertConfig, BertModel
# import pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR as LinearWarmupCosineAnnealingLR

from torch.optim.lr_scheduler import _LRScheduler
from model.resnet import ResNet, BasicBlock

from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict




# from fa import BertModel as fa_BertModel
# from fa import BertConfig as fa_BertConfig

import scipy
import uuid

from aug.aug import aug_near_feautee_change, aug_near_mix, aug_randn

# import lightning_lite
import manifolds

# from kornia.augmentation import Normalize, RandomGaussianBlur, RandomSolarize, RandomGrayscale, RandomResizedCrop, RandomCrop, ColorJitter, RandomChannelShuffle, RandomHorizontalFlip, RandomThinPlateSpline
import math

new_uuid = str(uuid.uuid4())

def muti_gumbel(logits, tau= 1, hard= False, eps= 1e-10, dim = -1, top_N=10, num_use_moe=10):
    
    mask_list = []
    mask_soft_list = []
    for i in range(num_use_moe):
        mask_soft, mask = gumbel_softmax_topN(logits[:,i,:], tau=tau, hard=hard, eps=eps, dim=dim, top_N=top_N)
        mask_list.append(mask)
        mask_soft_list.append(mask_soft)
    return torch.stack(mask_list, dim=1), torch.stack(mask_soft_list, dim=1)

def gumbel_softmax_topN(logits, tau= 1, hard= False, eps= 1e-10, dim = -1, top_N=10):

    # if has_torch_function_unary(logits):
    #     return handle_torch_function(gumbel_softmax, (logits,), logits, tau=tau, hard=hard, eps=eps, dim=dim)
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.topk(k=top_N, dim=dim)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return y_soft, ret



class CosineAnnealingSchedule(_LRScheduler):
    """Cosine annealing with warmup."""

    def __init__(
        self, opt, final_lr=0, n_epochs=1000, warmup_epochs=10, warmup_lr=0
    ):
        self.opt = opt
        self.optimizer = self.opt
        self.base_lr = base_lr = opt.defaults["lr"]
        self.final_lr = final_lr
        self.n_epochs = n_epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr

        # increase the number by one since we initialize the optimizer
        # before the first step (so the lr is set to 0 in the case of
        # warmups).  So we start counting at 1, basically.
        decay_epochs = 1 + n_epochs - warmup_epochs
        self.decay_epochs = decay_epochs

        warmup_schedule = np.linspace(warmup_lr, base_lr, warmup_epochs)
        decay_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
            1 + np.cos(np.pi * np.arange(decay_epochs) / decay_epochs)
        )
        self.lr_schedule = np.hstack((warmup_schedule, decay_schedule))

        self._last_lr = self.lr_schedule[0]
        self.cur_epoch = 0

        self.init_opt()

    def init_opt(self):
        self.step()
        # self.set_epoch(0)

    def get_lr(self):
        return self.lr_schedule[self.cur_epoch]

    def step(self):
        for param_group in self.opt.param_groups:
            lr = param_group["lr"] = self.get_lr()

        self.cur_epoch += 1
        self._last_lr = lr
        return lr

    def set_epoch(self, epoch):
        self.cur_epoch = epoch


class NN_FCBNRL_MM(nn.Module):
    def __init__(self, in_dim, out_dim, channel=8, use_RL=True, use_BN=True, use_DO=True):
        super(NN_FCBNRL_MM, self).__init__()
        m_l = []
        m_l.append(nn.Linear(in_dim, out_dim))
        if use_DO:
            m_l.append(nn.Dropout(p=0.02))
        if use_BN:
            m_l.append(nn.BatchNorm1d(out_dim))
        if use_RL:
            m_l.append(nn.LeakyReLU(0.1))
        
        self.block = nn.Sequential(*m_l)

    def forward(self, x):
        # if self.block[0].weight.shape[0] == self.block[0].weight.shape[1]:
        #     return self.block(x) + x
        # else:
        return self.block(x)

class CustomBertModel(nn.Module):
    def __init__(self, config):
        super(CustomBertModel, self).__init__()
        self.bert = BertModel(config)
        # 为了直接访问BertEmbeddings和BertEncoder，我们在这里不直接使用self.bert
        self.embeddings = self.bert.embeddings
        self.encoder = self.bert.encoder
        # self.pooler = self.bert.pooler
        self.fc = nn.Sequential(
            NN_FCBNRL_MM(config.hidden_size, 1000),
            NN_FCBNRL_MM(1000, 1000),
            NN_FCBNRL_MM(1000, config.hidden_size),
        )
        self.pooler = self.bert.pooler

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, mask=None):
        # Step 1: 使用Bert的嵌入层
        # if inputs_embeds is None:
        #     inputs_embeds = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        inputs_embeds = input_ids
        # if mask is not None:
        # import pdb; pdb.set_trace()
        # inputs_embeds = inputs_embeds * mask.reshape(-1, 1, inputs_embeds.shape[-1])

        # inputs_embeds_fc = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
        # fc_out = self.fc(inputs_embeds_fc).reshape(inputs_embeds.shape[0], inputs_embeds.shape[1], -1)

        encoder_outputs = self.encoder(
            hidden_states=inputs_embeds,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        # 取编码器的输出
        sequence_output = encoder_outputs['last_hidden_state'] # + fc_out
        # sequence_output = fc_out
        
        pooled_output = self.pooler(sequence_output)
        
        # 返回编码器的输出和pooled输出，根据需要返回其他输出
        return pooled_output


class TransformerEncoder(nn.Module):
    def __init__(
        self, 
        num_layers=2, 
        num_attention_heads=6, 
        hidden_size=240, 
        intermediate_size=300, 
        max_position_embeddings=784, 
        num_input_dim=784,
        # out_dim=50,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        num_use_moe=10,
        use_moe=True,
        ):
        super(TransformerEncoder, self).__init__()
        self.use_moe = use_moe
        # self.num_muti_mask = num_muti_mask
        # config = BertConfig(
        #     vocab_size=num_input_dim+2,  # 词汇表大小
        #     # embedding_size=trans_embedding_size,  # 嵌入层大小
        #     hidden_size=hidden_size,  # 隐藏层大小
        #     num_hidden_layers=num_layers,  # 隐藏层的数量
        #     num_attention_heads=num_attention_heads,  # 注意力头的数量
        #     intermediate_size=intermediate_size,  # 前馈网络的大小
        #     max_position_embeddings=max_position_embeddings,  # 最大序列长度
        #     hidden_dropout_prob=hidden_dropout_prob,  # 隐藏层的Dropout概率
        #     attention_probs_dropout_prob=attention_probs_dropout_prob,  # 注意力层的Dropout概率
        #     # num_hidden_layers=num_layers_Transformer,
        # )
        # self.enc = CustomBertModel(config)
        
        if num_input_dim == 3072:
            nn_type = 'resnet'
            print('use resnet')
        else:
            nn_type = 'nn'
            print('use nn')
            
        # else:
        if self.use_moe:
            self.enc = torch.nn.ModuleList([
                self.network_single(
                    num_input_dim,
                    hidden_size,
                    num_layers,
                    nn_type=nn_type,
                    ) for i in range(num_use_moe) 
                ])
        else:
            self.enc = self.network_single(
                num_input_dim, 
                hidden_size, 
                num_layers,
                nn_type=nn_type,
                )
        
        self.fc = nn.Sequential(
            # NN_FCBNRL_MM(hidden_size, hidden_size),
            # NN_FCBNRL_MM(hidden_size, hidden_size),
            NN_FCBNRL_MM(hidden_size, num_input_dim, use_RL=False),
        )
    
    def network_single(self, num_input_dim, hidden_size, num_layers, nn_type='nn'):
        
        if nn_type=='resnet':
            enc = ResNet(BasicBlock, [2, 2, 2, 2], 3)
        else:
            enc_list = []
            enc_list.append(NN_FCBNRL_MM(num_input_dim, hidden_size))
            for i in range(num_layers):
                enc_list.append(
                    NN_FCBNRL_MM(hidden_size, hidden_size)
                    )
            enc_list.append(NN_FCBNRL_MM(hidden_size, hidden_size, use_RL=False))
            enc = nn.Sequential(*enc_list)
        return enc
    
    def forward(self, input_x):
        if self.use_moe:
            emb_all = [self.fc(enc(input_x[:,i,:])) for i, enc in enumerate(self.enc)]
            emb = torch.stack(emb_all, dim=1)
        else:
            emb = self.fc(self.enc(input_x))
        return emb


class DMTEVT_model(LightningModule):
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
        tau=20,
        T_num_layers=2,
        T_num_attention_heads=6,
        T_hidden_size=240,
        T_intermediate_size=300,
        T_hidden_dropout_prob=0.1,
        T_attention_probs_dropout_prob=0.1,
        ckpt_path=None,
        use_orthogonal=False,
        emb_lat_w=0.5,
        num_use_moe=1,
        vis_dim=2,
        trans_out_dim=50,
        max_epochs=600,
        v_latent=0.01,
        n_neg_sample=4,
        **kwargs,
    ):
        super().__init__()

        # Set our init args as class attributes
        self.setup_bool_zzl = False
        # self.learning_rate = learning_rate
        self.save_hyperparameters()

        num_input_dim = self.hparams.num_input_dim
        
        # self.enc = self.InitNetworkMLP(NS=[num_input_dim, 500]+[200]*10+[50])
        self.enc = TransformerEncoder(
            num_layers=T_num_layers, 
            num_attention_heads=T_num_attention_heads, 
            hidden_size=T_hidden_size,
            intermediate_size=T_intermediate_size,
            max_position_embeddings=20,
            num_input_dim=num_input_dim,
            hidden_dropout_prob=T_hidden_dropout_prob,
            attention_probs_dropout_prob=T_attention_probs_dropout_prob,
            # out_dim=trans_out_dim,
            num_use_moe=num_use_moe,
        )
        
        # self.dec = self.InitNetworkMLP(NS=[vis_dim, 500, 500, num_input_dim], last_relu=False)
        self.vis = self.InitNetworkMLP(NS=[num_input_dim, 500, vis_dim], last_relu=False)
        self.exp = self.InitNetworkMLP(NS=[vis_dim+vis_dim, 500, 500, num_input_dim*num_use_moe])
        
        self.vis_dictionary = torch.randn((num_train_data, vis_dim)).to(self.device)

        if ckpt_path is not None:
            print('load ckpt from:', ckpt_path)
            # if not os.path.exists(ckpt_path+'/all.ckpt'):
            #     convert_zero_checkpoint_to_fp32_state_dict(ckpt_path, ckpt_path+'/all.ckpt')
            # self.load_state_dict(torch.load(ckpt_path)['state_dict'])
            self.load_state_dict(torch.load(ckpt_path))

        # import pdb; pdb.set_trace()

    def InitNetworkMLP(self, NS, last_relu=True, use_DO=True, use_BN=True, use_RL=True):

        m_l = []
        for i in range(len(NS) - 1):
            if i == len(NS) - 2 and not last_relu:
                m_l.append(NN_FCBNRL_MM(NS[i], NS[i + 1], use_RL=False, use_DO=use_DO, use_BN=use_BN))
            else:
                m_l.append(NN_FCBNRL_MM(NS[i], NS[i + 1], use_RL=True, use_DO=use_DO, use_BN=use_BN))
            
        model_pat = nn.Sequential(*m_l)

        return model_pat

    def _DistanceSquared(self, x, y=None, metric="euclidean"):
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
                dist[torch.eye(dist.shape[0]) == 1] = 1e-12
        return dist

    def _CalGamma(self, v):
        a = scipy.special.gamma((v + 1) / 2)
        b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
        out = a / b
        return out

    def _Similarity(self, dist, sigma=0.3):
        dist_rho = dist
        dist_rho[dist_rho < 0] = 0
        Pij = torch.exp(-dist / (2 * sigma ** 2))
        return Pij

    def _Similarity_old(self, dist, gamma, v=100, h=1, pow=2):
        dist_rho = dist

        dist_rho[dist_rho < 0] = 0
        Pij = (
            gamma
            * torch.tensor(2 * 3.14)
            * gamma
            * torch.pow((1 + dist_rho / v), exponent=-1 * (v + 1))
        )
        return Pij


    # def LossManifold(
    #     self,
    #     input_data,
    #     latent_data,
    #     sigma=0.03,
    #     w_nb=500,
    #     w_fp=3,
    #     high_tok_rank=1,
    #     n_select_each_sample=4,
    # ):

    #     # data_1 = input_data[: input_data.shape[0] // 2]
        
    #     # dis_P = self._DistanceSquared(data_1)
    #     latent_data_1 = latent_data[: input_data.shape[0] // 2]
    #     latent_data_2 = latent_data[(input_data.shape[0] // 2):]
    #     NB = torch.pairwise_distance(latent_data_1, latent_data_2, p=2, keepdim=False) + 1
    #     loss_NB = (NB/(10+NB)).mean()
        

    #     loss_fp_list = []
    #     for i in range(n_select_each_sample):
    #         latent_data_2_rand = latent_data_2[torch.randperm(latent_data_2.shape[0])]
    #         FP = torch.pairwise_distance(latent_data_1, latent_data_2_rand, p=2, keepdim=False) + 1
    #         loss_fp_list.append((1/(1+FP)).mean())

    #     loss_fp = torch.stack(loss_fp_list).mean()

    #     return loss_fp*w_fp, loss_NB*w_nb

    def t_distribution_similarity(self, distance_matrix, df):
        """
        计算 t 分布相似度矩阵
        :param distance_matrix: 输入距离矩阵，形状为 [N, N]
        :param df: 自由度
        :return: 相似度矩阵，形状为 [N, N]
        """
        # 计算 t 分布的相似度矩阵
        distance_matrix = distance_matrix+1e-6
        numerator = (1 + distance_matrix ** 2 / df) ** (-(df + 1) / 2)
        denominator = torch.sum(numerator, dim=1, keepdim=True) - torch.diagonal(numerator, 0).unsqueeze(1)
        similarity_matrix = numerator / denominator
        
        return similarity_matrix


    def LossManifold(
        self,
        latent_data,
        temperature=1,
        exaggeration=1,
        nu=0.1,
    ):
        batch_size = latent_data.shape[0] // 2        
        
        features_a = latent_data[:batch_size]
        features_b = latent_data[batch_size:]

        dis_aa = torch.cdist(features_a, features_a) * temperature
        dis_bb = torch.cdist(features_b, features_b) * temperature
        dis_ab = torch.cdist(features_a, features_b) * temperature

        sim_aa = self.t_distribution_similarity(dis_aa, df=nu)
        sim_bb = self.t_distribution_similarity(dis_bb, df=nu)
        sim_ab = self.t_distribution_similarity(dis_ab, df=nu)
        
 
        tempered_alignment = (torch.diagonal(sim_ab).log()).mean()

        # exclude self inner product
        self_mask = torch.eye(batch_size, dtype=bool, device=sim_aa.device)
        sim_aa.masked_fill_(self_mask, 0.0)
        sim_bb.masked_fill_(self_mask, 0.0)

        logsumexp_1 = torch.hstack((sim_ab.T, sim_bb)).sum(1).log_().mean()
        logsumexp_2 = torch.hstack((sim_aa, sim_ab)).sum(1).log_().mean()

        raw_uniformity = logsumexp_1 + logsumexp_2

        loss = -(exaggeration * tempered_alignment - raw_uniformity / 2)

        return loss

    def _TwowaydivergenceLoss(self, P_, Q_, select=None):
        EPS = 1e-5
        losssum1 = P_ * torch.log(Q_ + EPS)
        losssum2 = (1 - P_) * torch.log(1 - Q_ + EPS)
        losssum = -1 * (losssum1 + losssum2)
        return losssum.mean()


    def oth_loss(self, features):
        
        batch_size, num_features, feature_dim = features.shape
        # import pdb; pdb.set_trace()
        features_sample = features[:100]
        # Flatten batch and feature dimensions
        
        # 组内
        features_sample_intra = features_sample.permute(1, 0, 2).mean(1)
        distance_matrix = torch.cdist(features_sample_intra, features_sample_intra, p=2)/10
        similarity_matrix = self.t_distribution_similarity(distance_matrix, 1)
        orthogonality_loss = torch.mean(similarity_matrix)

        # 组间
        # import pdb; pdb.set_trace()
        features_sample_inter = features_sample.permute(1, 0, 2)
        for i in range(features_sample_inter.shape[0]):
            vectors = features_sample_inter[i]
            norms = vectors.norm(dim=1, keepdim=True)
            # norms = torch.where(norms == 0, torch.tensor(1.0, device=vectors.device), norms)  # 避免除以0
            norm_vectors = vectors / norms
            similarity_matrix = torch.mm(norm_vectors, norm_vectors.t())
            similarity_matrix = torch.clamp(similarity_matrix, min=-1.0, max=1.0)
            sim_mean = 1-similarity_matrix.mean()
            
            if i == 0:
                loss_in = sim_mean
            else:
                loss_in += sim_mean
        # distance_matrix = torch.cdist(features_sample, features_sample, p=2)/10
        # similarity_matrix = self.t_distribution_similarity(distance_matrix, 1)
        # orthogonality_loss = torch.mean(similarity_matrix)

        loss = loss_in/features_sample_inter.shape[0] - orthogonality_loss
        return loss


    def batch_orthogonal_loss(self, batch_u, batch_v):
        # 计算批量向量的点积，假设batch_u和batch_v的形状为[N, D]，其中N是批量大小，D是向量维度
        cosine_sims = F.cosine_similarity(batch_u, batch_v, dim=1)
        # 返回余弦相似度平方的平均值作为损失
        loss = torch.mean(cosine_sims ** 2)
        return loss

    def batch_patten_loss(self, feature_tra):
        
        feature_tra = feature_tra + torch.randn_like(feature_tra)*0.001*feature_tra.std()
        
        # feature_tra = feature_tra + torch.randn_like(feature_tra) * 0.001 * feature_tra.std()
        # 处理每个样本中的所有向量
        norms = feature_tra.norm(dim=2, keepdim=True)
        norms = torch.where(norms == 0, torch.tensor(1.0, device=feature_tra.device), norms)  # 避免除以0
        norm_vectors = feature_tra / norms

        # 计算每个样本的相似度矩阵
        similarity_matrices = torch.einsum('bij,bkj->bik', norm_vectors, norm_vectors)
        similarity_matrices = torch.clamp(similarity_matrices, min=-1.0, max=1.0)

        # 计算平均相似度并累积损失
        sim_means = 1 - similarity_matrices.mean(dim=(1, 2))
        loss_in = sim_means.mean()

        # 处理每组中的所有向量
        norm_vectors = norm_vectors.permute(1, 0, 2)  # 将形状从 (batch_size, num_vectors_per_group, vector_dim) 变为 (num_vectors_per_group, batch_size, vector_dim)
        similarity_matrices = torch.einsum('bij,bkj->bik', norm_vectors, norm_vectors)
        similarity_matrices = torch.clamp(similarity_matrices, min=-1.0, max=1.0)

        # 计算平均相似度并累积损失
        sim_means = 1 - similarity_matrices.mean(dim=(1, 2))
        loss_cr = sim_means.mean()
        
        return loss_in - loss_cr
        

    def forward(self, x, tau=100.0):
        
        batch_size = x.shape[0] // 2
        
        if len(x.shape) == 2:
            num_select = int(x.shape[1]*self.hparams.sample_rate_feature)
            if len(x.shape) == 2:
                expanded_x = x.unsqueeze(1).expand(-1, self.hparams.num_use_moe, -1)
            elif len(x.shape) == 4:
                expanded_x = x.unsqueeze(1).expand(-1, self.hparams.num_use_moe, -1, -1, -1)
            
            expanded_enc_out = self.enc(expanded_x)
            expanded_cond = self.vis(expanded_enc_out.reshape(-1, expanded_enc_out.shape[-1]))
            cond = expanded_cond.reshape(expanded_enc_out.shape[0], expanded_enc_out.shape[1], -1)
            cond = cond[:,0,:]
            weight = self.get_weight(cond=cond)
            self.mask, self.soft_mask = muti_gumbel(
                weight, 
                tau=tau, 
                hard=True, 
                # hard=False, 
                top_N=num_select, 
                num_use_moe=self.hparams.num_use_moe
                )
            
            if self.mask.shape[0] != x.shape[0]:
                self.mask = torch.cat([self.mask, self.mask])
            
            if len(x.shape) == 2:
                x_masked = torch.einsum('bik,bk->bik', self.mask, x)
            elif len(x.shape) == 4:
                _, num_channel, num_x, num_y = x.shape
                x = x.reshape((x.shape[0], -1))
                x_masked = torch.einsum('bik,bk->bik', self.mask, x)
                x_masked = x_masked.reshape((batch_size*2, self.hparams.num_use_moe, num_channel, num_x, num_y))
        else:
            # import pdb; pdb.set_trace()
            x_masked = x.unsqueeze(1).expand(-1, self.hparams.num_use_moe, -1, -1, -1)
            weight = x_masked
        
        lat_higt_dim_out = self.enc(x_masked)
        lat_high_dim = lat_higt_dim_out.reshape((-1, lat_higt_dim_out.shape[-1]))
        lat_vis = self.vis(lat_high_dim)
        # import pdb; pdb.set_trace()
        lat_vis_out = lat_vis.reshape((batch_size*2, -1, lat_vis.shape[-1]))
        
        return weight, lat_higt_dim_out, lat_vis_out
    
    
    def get_weight(self, cond):
        
        # import pdb; pdb.set_trace()
        # print('cond', cond.shape, cond.device)
        rand = torch.randn(cond.shape[0], self.hparams.vis_dim).to(cond.device)
        input_data = torch.cat([cond, rand], dim=1)
        w = self.exp(input_data).reshape((cond.shape[0], self.hparams.num_use_moe, -1))
        weight = F.tanh(w)*10
        return weight
    
    
    def get_tau(self, epoch, total_epochs=900, tau_start=100, tau_end=1.001):
        """
        计算指定 epoch 的 tau 值。
        
        :param epoch: 当前的 epoch 数 (从0开始)
        :param total_epochs: 总的 epoch 数
        :param tau_start: 开始的 tau 值
        :param tau_end: 结束的 tau 值
        :return: 计算得到的 tau 值
        """
        if epoch >= total_epochs:
            return tau_end
        else:    
            return tau_start * (tau_end / tau_start) ** (epoch / (total_epochs-1))

    def training_step(self, batch, batch_idx):
        data_input_item = batch['data_input_item']
        data_input_aug = batch['data_input_aug']
        index = batch['index']

        data_input_item = torch.cat([data_input_item, data_input_aug])
        x_masked, lat_high_dim_exp, lat_vis_exp = self(
            data_input_item, 
            tau=self.hparams.tau,
            )
        lat_high_dim = lat_high_dim_exp.mean(dim=1)
        lat_vis = lat_vis_exp.mean(dim=1)
        
        
        if self.hparams.use_orthogonal:
            orthogonal_loss = self.batch_patten_loss(x_masked[:200].permute(1, 2, 0))
        else:
            orthogonal_loss = 0 # self.batch_patten_loss(x_masked[:200].permute(1, 2, 0))
        
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
        
        loss_all = self.hparams.emb_lat_w * loss_emb + (1-self.hparams.emb_lat_w) * loss_lat + orthogonal_loss*0.01#+ loss_mse*self.hparams.weight_mse # + 0.1*orthogonal_loss#+ kloss# + mlm_loss / 10

        self.log('loss_all', loss_all, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('loss_emb', loss_emb, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('loss_lat', loss_lat, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('orthogonal_loss', orthogonal_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('lr', float(self.trainer.optimizers[0].param_groups[0]["lr"]), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss_all

    def validation_step(self, batch, batch_idx, test=False, dataloader_idx=0):
        # data_input_item, data_input_aug, label, index, = batch
        if dataloader_idx == 0:
            data_input_item = batch['data_input_item']
            data_input_aug = batch['data_input_aug']
            index = batch['index']

            if self.vis_dictionary.device != self.device:
                self.vis_dictionary = self.vis_dictionary.to(self.device)
                # match the dtype of vis_dictionary

            # cond = self.vis_dictionary[index.to(self.vis_dictionary.device)]
            # print('cond', cond.shape)
            x_masked, lat_high_dim_exp, lat_vis_exp = self(
                data_input_item,
                tau=self.hparams.tau,
                )
            lat_high_dim = lat_high_dim_exp.mean(dim=1)
            lat_vis = lat_vis_exp.mean(dim=1)
            # import pdb; pdb.set_trace()
            
            # import pdb; pdb.set_trace()
            self.validation_origin_input = data_input_item
            self.validation_step_outputs_high = lat_high_dim
            self.validation_step_outputs_vis = lat_vis
            self.validation_step_lat_vis_exp = lat_vis_exp
    
    def test_step(self, batch, batch_idx):
        
        data_input_item = batch['data_input_item']
        data_input_aug = batch['data_input_aug']
        label = batch['label']

        if self.vis_dictionary.device != self.device:
            self.vis_dictionary = self.vis_dictionary.to(self.device)
            # match the dtype of vis_dictionary

        # cond_rand = torch.randn(
        #     (data_input_item.shape[0], self.hparams.vis_dim)
        #     ).to(self.vis_dictionary.device)
        # lat_high_dim, cond = self(
        #     data_input_item,
        #     # cond = cond_rand.detach().to(self.device)
        #     )
        
        # cond = lat_vis
        # lat_high_dim, cond = self(
        #     data_input_item,
        #     # cond = cond.detach().to(self.device)
        #     )
        
        x_masked, lat_high_dim, lat_vis = self(
            data_input_item,
            # cond = cond.detach().to(self.device)
            )

        self.test_step_outputs_high = lat_high_dim
        self.test_step_outputs_vis = lat_vis
        self.test_step_outputs_label = label

        # import pdb; pdb.set_trace()
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            weight_decay=self.hparams.weight_decay, 
            lr=self.hparams.lr
            )
        lrsched = CosineAnnealingSchedule(
            optimizer, n_epochs=self.hparams.max_epochs , warmup_epochs=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lrsched,
                "interval": "epoch",
            },  # interval "step" for batch update
        }

