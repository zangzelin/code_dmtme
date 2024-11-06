from lightning.pytorch.callbacks import Callback
import matplotlib.pyplot as plt
import torch
import wandb
import eval.eval_core_base as ecb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.svm import SVC
import numpy as np

import matplotlib.pyplot as plt

class MaskExpCallBack(Callback):
    def __init__(self, inter=10, only_val=False, *args, **kwargs):
        super().__init__()
        self.inter = inter
        self.only_val = only_val
        self.plot_sample_train = 0
        self.plot_sample_test = 0
        self.val_mask_list = []
        
        self.val_ori_single_image_list = []


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # preds = pl_module.step_last_outputs_val  # 假设模型的输出字典中包含预测结果
        # data_input_item, data_input_aug, label, index, = batch
        if dataloader_idx==0:
            data_input_item = batch["data_input_item"]
            
            mask_this_batch = pl_module.mask
            
            self.val_mask_list.append(mask_this_batch)
            self.val_ori_single_image_list.append(data_input_item)

    def plot_statistic(self, gathered_mask, ori_img, trainer, num_muti_mask=1):        
        # plot mean with heatmap
        
        for mask_index in range(num_muti_mask):
            
            gathered_mask_single = gathered_mask[:, mask_index, :]
            # import pdb; pdb.set_trace()
            if gathered_mask_single.shape[1] == 784:
                mask_mean = np.mean(gathered_mask_single, axis=0).reshape(28, 28)
                plt.figure(figsize=(5, 5))
                plt.imshow(mask_mean, cmap='Reds')  # Changing the colormap to a gradient from white to red
                trainer.logger.experiment.log({f"maskmean/mask_mean_mask{mask_index}": [wandb.Image(plt)]})
                plt.close()
            else:
                # rise an error
                print ("Not a 28x28 mask")

    
    def plot_single_image_and_mask(self, gathered_mask_all, gathered_ori_img, trainer, num_muti_mask=1, num_sample=10):

        # import pdb; pdb.set_trace()

        plt.figure(figsize=(50, 60))
        for i in range(num_sample):
            plt.subplot(num_sample, num_muti_mask+1, i*(num_muti_mask+1)+1)
            plt.imshow(gathered_ori_img[i].reshape(28, 28), cmap='gray')
            # plt.colorbar()
            for mask_index in range(num_muti_mask):
                gathered_mask = gathered_mask_all[:, mask_index, :]
                plt.subplot(num_sample, num_muti_mask+1, i*(num_muti_mask+1)+2+mask_index)
                plt.imshow(gathered_mask[i].reshape(28, 28), cmap='viridis')
                # plt.colorbar()
        trainer.logger.experiment.log({f"masksingle/mask_single_mask{mask_index}": [wandb.Image(plt)]})
        plt.close()

    def on_validation_epoch_end(self, trainer, pl_module):
        
        # if trainer.current_epoch % self.inter == 0:
        mask = torch.cat(self.val_mask_list).to(pl_module.device)
        ori_img = torch.cat(self.val_ori_single_image_list).to(pl_module.device)
        
        if len(mask.shape) == 3:
            num_muti_mask = mask.shape[1]
        else:
            num_muti_mask = 1
        
        gathered_ori_img = ori_img.cpu().detach().numpy()
        gathered_mask = mask.cpu().detach().numpy()
        # gathered_mask= trainer.strategy.all_gather(mask).cpu().detach().numpy()
        # gathered_ori_img = trainer.strategy.all_gather(ori_img).cpu().detach().numpy()
        
        # import pdb; pdb.set_trace()
        # gathered_mask = gathered_mask.reshape(-1, gathered_mask.shape[-1])
        # gathered_ori_img = gathered_ori_img.reshape(-1, gathered_ori_img.shape[-1])
        
        print('current_epoch:', trainer.current_epoch)
        if (self.only_val or trainer.current_epoch > 1) and (trainer.current_epoch+1) % self.inter == 0:
        # import pdb; pdb.set_trace()
            # pass
            if trainer.is_global_zero:  # 检查是否为 rank 0 的进程    
                self.plot_statistic(gathered_mask, ori_img, trainer, num_muti_mask)
                # self.plot_single_image_and_mask(gathered_mask, gathered_ori_img, trainer, num_muti_mask)
                # shape (N, num_features)

        self.val_ori_single_image_list = []
        self.val_mask_list = []