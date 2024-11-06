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

# from fa import BertModel as fa_BertModel
# from fa import BertConfig as fa_BertConfig

import scipy
import uuid
import matplotlib.pyplot as plt

import eval.eval_core as ec
import eval.eval_core_base as ecb

from aug.aug import aug_near_feautee_change, aug_near_mix, aug_randn
from dataloader import data_rankseq as data_base

# import lightning_lite
import manifolds

# from kornia.augmentation import Normalize, RandomGaussianBlur, RandomSolarize, RandomGrayscale, RandomResizedCrop, RandomCrop, ColorJitter, RandomChannelShuffle, RandomHorizontalFlip, RandomThinPlateSpline
import math

new_uuid = str(uuid.uuid4())

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
    return ret

class NN_FCBNRL_MM(nn.Module):
    def __init__(self, in_dim, out_dim, channel=8, use_RL=True):
        super(NN_FCBNRL_MM, self).__init__()
        m_l = []
        m_l.append(
            nn.Linear(
                in_dim,
                out_dim,
            )
        )
        m_l.append(nn.Dropout(p=0.02))
        m_l.append(nn.BatchNorm1d(out_dim))
        if use_RL:
            m_l.append(nn.LeakyReLU(0.1))
        self.block = nn.Sequential(*m_l)

    def forward(self, x):
        return self.block(x)



class DMTEVT_Vis(nn.Module):
    def __init__(
        self,
        l_token,
        l_token2,
        data_name,
        transformer2_indim,
        laten_down,
        num_layers_Transformer,
        num_input_dim,
    ):
        super(DMTEVT_Vis, self).__init__()
        self.model_down = self.InitNetworkMLP(
            l_token=l_token,
            l_token_2=l_token2,
            data_name=data_name,
            transformer2_indim=transformer2_indim,
            laten_down=laten_down,
            num_layers_Transformer=num_layers_Transformer,
            num_input_dim=num_input_dim,
        )

    def forward(self, lat_high_dim):
        lat_vis = self.model_down(lat_high_dim)
        return lat_vis

    def InitNetworkMLP(
        self,
        l_token=50,
        l_token_2=100,
        data_name="mnist",
        transformer2_indim=3750,
        laten_down=500,
        num_layers_Transformer=1,
        num_input_dim=64,
    ):
        if "Cifar" not in data_name:
            m_b = []
            m_b.append(NN_FCBNRL_MM(l_token_2, laten_down))
            m_b.append(NN_FCBNRL_MM(laten_down, laten_down))
            m_b.append(NN_FCBNRL_MM(laten_down, 2, use_RL=False))
            m_b.append(nn.BatchNorm1d(2))
            model_down = nn.Sequential(*m_b)
        else:
            m_b = []
            m_b.append(NN_FCBNRL_MM(l_token_2, laten_down))
            m_b.append(NN_FCBNRL_MM(laten_down, 2, use_RL=False))
            model_down = nn.Sequential(*m_b)

        return model_down


class DMTEVT_Decoder(nn.Module):
    def __init__(
        self,
        l_token,
        l_token2,
        data_name,
        transformer2_indim,
        laten_down,
        num_layers_Transformer,
        num_input_dim,
    ):
        super(DMTEVT_Decoder, self).__init__()
        self.model_decoder = self.InitNetworkMLP(
            l_token=l_token,
            l_token_2=l_token2,
            data_name=data_name,
            transformer2_indim=transformer2_indim,
            laten_down=laten_down,
            num_layers_Transformer=num_layers_Transformer,
            num_input_dim=num_input_dim,
        )

    def forward(self, x):
        return self.model_decoder(x)

    def InitNetworkMLP(
        self,
        l_token=50,
        l_token_2=100,
        data_name="mnist",
        transformer2_indim=3750,
        laten_down=500,
        num_layers_Transformer=1,
        num_input_dim=64,
    ):
        m_decoder = []
        m_decoder.append(NN_FCBNRL_MM(l_token_2, 500))
        m_decoder.append(NN_FCBNRL_MM(500, 500))
        m_decoder.append(NN_FCBNRL_MM(500, num_input_dim, use_RL=False))
        # m_decoder.append(nn.Sigmoid())
        model_decoder = nn.Sequential(*m_decoder)

        return model_decoder


class DMTEVT_model(LightningModule):
    def __init__(
        self,
        lr=0.005,
        nu=0.01,
        data_name="mnist",
        batch_size=900,
        max_epochs=1000,
        class_num=7,
        steps=20000,
        sample_rate_feature=0.6,
        num_fea_aim=500,
        l_token=50,
        l_token_2=50,
        num_layers_Transformer=1,
        num_latent_dim=2,
        num_input_dim=64,
        laten_down=500,
        weight_decay=0.0001,
        kmean_scale=0.01,
        loss_rec_weight=1.0,
        preprocess_epoch=100,
        transformer2_indim=3750,
        marker_size=2,
        sample_len=500,
        trans_embedding_size=64,
        num_attention_heads=6,
        w=1,
        intermediate_size=128,
        uselabel=False,
        top_k_1=3,
        top_k_2=500,
        num_fea_per_pat=50,
        **kwargs,
    ):
        super().__init__()

        # Set our init args as class attributes
        self.setup_bool_zzl = False
        # self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.t = 0.1
        self.alpha = None
        self.stop = False
        self.bestval = 0
        self.aim_cluster = None
        self.importance = None
        self.wandb_logs = {}
        self.mse = torch.nn.MSELoss()  #
        self.ce = torch.nn.CrossEntropyLoss()

        self.enc = self.InitNetworkMLP(
            NetworkStructure_1=[-1, 500] + [500] * 2,
            NetworkStructure_2=[-1, 500, 80],
        )
        
        self.dec = DMTEVT_Decoder(
            l_token=self.hparams.l_token,
            l_token2=self.hparams.l_token_2,
            data_name=self.hparams.data_name,
            transformer2_indim=self.hparams.transformer2_indim,
            laten_down=self.hparams.laten_down,
            num_layers_Transformer=self.hparams.num_layers_Transformer,
            num_input_dim=self.hparams.num_input_dim,
        )
        self.vis = DMTEVT_Vis(
            l_token=self.hparams.l_token,
            l_token2=self.hparams.l_token_2,
            data_name=self.hparams.data_name,
            transformer2_indim=self.hparams.transformer2_indim,
            laten_down=self.hparams.laten_down,
            num_layers_Transformer=self.hparams.num_layers_Transformer,
            num_input_dim=self.hparams.num_input_dim,
        )
        
        self.exp = nn.Sequential(
            NN_FCBNRL_MM(2, 500,),
            NN_FCBNRL_MM(500, 500,),
            NN_FCBNRL_MM(500, 784,),
        )

    def InitNetworkMLP(self, NetworkStructure_1, NetworkStructure_2):

        num_fea_per_pat = self.hparams.num_fea_per_pat
        struc_model_pat = (
            [784]
            + NetworkStructure_1[1:]
            + [num_fea_per_pat]
        )
        struc_model_b = NetworkStructure_2 + [self.hparams.num_latent_dim]
        struc_model_b[0] = num_fea_per_pat

        m_l = []
        for i in range(len(struc_model_pat) - 1):
            m_l.append(
                NN_FCBNRL_MM(struc_model_pat[i],struc_model_pat[i + 1],)
            )
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

    def _Similarity(self, dist, sigma=0.3, gamma=1, v=100, h=1, pow=2):
        dist_rho = dist
        dist_rho[dist_rho < 0] = 0
        Pij = torch.exp(-dist / (2 * sigma ** 2))
        return Pij

    def LossManifold(
        self,
        input_data,
        latent_data,
        v_latent,
        top_k_2=1000,
        top_k_1=3,
    ):

        data_1 = input_data[: input_data.shape[0] // 2]
        
        dis_P = self._DistanceSquared(data_1)
        latent_data_1 = latent_data[: input_data.shape[0] // 2]
        latent_data_2 = latent_data[(input_data.shape[0] // 2):]
        dis_Q_2 = self._DistanceSquared(latent_data_1, latent_data_2)
        Q_2 = self._Similarity(
            dist=dis_Q_2,
            gamma=self._CalGamma(v_latent),
            v=v_latent,
            sigma=0.05,
        )
        
        _, indices = torch.topk(dis_P, 3, dim=-1, largest=False)
        dis_P_topk_mask = torch.zeros_like(Q_2, dtype=torch.bool)
        dis_P_topk_mask.scatter_(dim=-1, index=indices, value=True)
        
        _, indices2 = torch.topk(dis_P, top_k_2, dim=-1, largest=False)
        dis_P_topk_mask2 = torch.zeros_like(Q_2, dtype=torch.bool)
        dis_P_topk_mask2.scatter_(dim=-1, index=indices2[:, top_k_1:], value=True)
        
        po = Q_2[dis_P_topk_mask.detach()]
        ne = Q_2[dis_P_topk_mask2.detach()]
        return ne.mean(), 1 - po.mean()

    def get_weight(self, cond):
        
        # if emb == None:
        #     lat = data
        #     lat1 = self.model_pat(lat)
        #     lat3 = lat1
        #     for i, m in enumerate(self.model_b):
        #         lat3 = m(lat3)
        #     emb = lat3.detach()
        rand_cond = torch.randn(cond.shape[0], 2).to(cond.device)
        w = self.exp(rand_cond)
        weight = F.tanh(w)*10
        return weight

    def forward(self, x, tau=100.0):

        if tau is not None:
            weight = self.get_weight(x)
            s_f = int(x.shape[1]*self.hparams.sample_rate_feature)
            self.mask = gumbel_softmax_topN(weight, tau=tau, hard=True, top_N=s_f)
            x = x * self.mask
        else:
            weight = self.get_weight(x)
            top_k_index = weight.topk(s_f, dim=1)[1]
            self.mask = torch.zeros_like(weight)
            self.mask = self.mask.scatter_(1, top_k_index, 1) > 0
            x = x * self.mask
        
        lat_high_dim = self.enc(x)
        lat_vis = self.vis(lat_high_dim)
        return lat_high_dim, lat_vis

    def training_step(self, batch, batch_idx):
        data_input_item, data_input_aug, label, index, = batch

        data_input_item = torch.cat([data_input_item, data_input_aug])
        lat_high_dim, lat_vis = self(data_input_item)
        
        loss_topo_ne, loss_topo_po = self.LossManifold(
            input_data=lat_high_dim.reshape(lat_high_dim.shape[0], -1),
            latent_data=lat_vis.reshape(lat_vis.shape[0], -1),
            v_latent=self.hparams.nu,
            top_k_1=self.hparams.top_k_1,
            top_k_2=self.hparams.top_k_2
        )
        
        loss_topo = loss_topo_ne + loss_topo_po # + loss_crossentropy
        loss_mse = ((self.dec(lat_high_dim)-data_input_item)**2).mean()
        loss_all = loss_topo + loss_mse*100 #+ kloss# + mlm_loss / 10


        self.log('loss_all', loss_all, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('loss_mse', loss_mse, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('loss_topo_ne', loss_topo_ne, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('loss_topo_po', loss_topo_po, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('lr', float(self.trainer.optimizers[0].param_groups[0]["lr"]), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # if self.current_epoch >= self.hparams.preprocess_epoch:
        # loss_all = loss_topo + loss_rec + kloss + mlm_loss / 100

        return loss_all

    def validation_step(self, batch, batch_idx, test=False):
        data_input_item, data_input_aug, label, index, = batch

        batch_size = index.shape[0]
        data = data_input_item
        # data_aug = self.transform(data_aug.permute(0,3,1,2)/255)
        lat_high_dim, lat_vis = self(data_input_item)
        data_recons = self.dec(lat_high_dim)
        
        self.validation_step_outputs_high = lat_high_dim
        self.validation_step_outputs_vis = lat_vis
        self.validation_step_outputs_recons = data_recons

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), weight_decay=self.hparams.weight_decay, lr=self.hparams.lr)
        self.scheduler = StepLR(optimizer, step_size=self.hparams.max_epochs // 10, gamma=0.8)
        return [optimizer], [self.scheduler]

