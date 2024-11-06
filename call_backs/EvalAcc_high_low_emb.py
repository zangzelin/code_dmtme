

from lightning.pytorch.callbacks import Callback
import matplotlib.pyplot as plt
import torch
import wandb
import eval.eval_core_base as ecb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.svm import SVC
import numpy as np
import os
from sklearn.manifold import TSNE
import plotly.graph_objects as go

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn import metrics
import eval.eval_core as ec
import eval.eval_core_base as ecb
from eval import evaluation
def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()

def euclidean_to_hyperbolic_matrix(u, c=0.5, min_norm = 1e-15):
    u = torch.tensor(u).float()
    
    u = 1.5 * ( u-u.mean(dim=0) )/u.std(dim=0)
    
    sqrt_c = c ** 0.5
    u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), min_norm)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return gamma_1.detach().numpy()



class EvalCallBack(Callback):
    def __init__(self, inter=10, dirpath='', fully_eval=False, dataset='', only_val=False, *args, **kwargs):
        super().__init__()
        self.inter = inter
        self.plot_sample_train = 0
        self.plot_sample_test = 0
        self.only_val = only_val
        # self.train_len_list = []
        # self.train_acc_list = []

        self.val_input= {'val1': [], 'val2': [], 'val3': []}
        self.val_high = {'val1': [], 'val2': [], 'val3': []}
        self.val_vis = {'val1': [], 'val2': [], 'val3': []}
        self.val_vis_exp = {'val1': [], 'val2': [], 'val3': []}
        self.val_label = {'val1': [], 'val2': [], 'val3': []}
        # self.val_recon = []

        # self.val_high_v2 = []
        # self.val_vis_v2 = []
        # self.val_label_v2 = []


        self.test_high = []
        self.test_vis = []
        self.test_recon = []
        self.test_label = []
        
        self.dirpath = dirpath
        self.dataset = dataset
        self.fully_eval = fully_eval
        self.best_acc = 0

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # preds = pl_module.step_last_outputs_val  # 假设模型的输出字典中包含预测结果
        # data_input_item, data_input_aug, label, index, = batch
        label = batch["label"]
        
        key_value = f"val{dataloader_idx+1}"
        
        self.val_input[key_value].append(pl_module.validation_origin_input)
        self.val_high[key_value].append(pl_module.validation_step_outputs_high)
        self.val_vis[key_value].append(pl_module.validation_step_outputs_vis)
        # self.val_vis_exp[key_value].append(pl_module.validation_step_lat_vis_exp)
        
        # self.validation_weight = pl_module.validation_weight
        # self.val_recon.append(pl_module.validation_step_outputs_recons)
        self.val_label[key_value].append(label)
        
        
        # self.val_high.append(pl_module.validation_step_outputs_high)
        # self.val_vis.append(pl_module.validation_step_outputs_vis)
        # # self.val_recon.append(pl_module.validation_step_outputs_recons)
        # self.val_label.append(label)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # preds = pl_module.step_last_outputs_val  # 假设模型的输出字典中包含预测结果
        # data_input_item, data_input_aug, label, index, = batch
        label = batch["label"]
        
        # self.test_high.append(pl_module.test_step_outputs_high)
        self.test_vis.append(pl_module.test_step_outputs_vis)
        # self.test_recon.append(pl_module.test_step_outputs_recons)
        self.test_label.append(label)

    def on_test_epoch_end(self, trainer, pl_module):
        
        test_vis = torch.cat(self.test_vis).cuda()
        test_label = torch.cat(self.test_label).cuda()
        gathered_val_vis = trainer.strategy.all_gather(test_vis).cpu().detach().numpy()
        gathered_val_label = trainer.strategy.all_gather(test_label).cpu().detach().numpy()
        
        gathered_val_vis = gathered_val_vis.reshape(-1, 2)
        gathered_val_label = gathered_val_label.reshape(-1)
        
        acc_mean = self.get_svc_acc(
            gathered_val_vis, 
            gathered_val_label, 
            trainer
            )
        print('gathered_val_vis', gathered_val_vis.shape)
        print('gathered_val_label', gathered_val_label.shape)
        print('acc_mean', acc_mean)        


    def plot_scatter(self, gathered_val_vis, gathered_val_label, trainer):
        fig = plt.figure(figsize=(10, 10))
        if gathered_val_vis.shape[0] >= 10000: 
            s=1
        else :
            s=3
        plt.scatter(
            gathered_val_vis[:, 0],
            gathered_val_vis[:, 1],
            c=gathered_val_label, 
            cmap='rainbow',
            s=s
        )
        return fig

    def plot_scatter_hyper(self, gathered_val_vis, gathered_val_label, trainer):
        
        gathered_val_vis_hy = euclidean_to_hyperbolic_matrix(gathered_val_vis)
        
        if gathered_val_vis.shape[0] >= 10000: 
            s=1
        else :
            s=3
        
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(
            gathered_val_vis_hy[:, 0],
            gathered_val_vis_hy[:, 1],
            c=gathered_val_label, 
            cmap='rainbow',
            s=s
        )
        return fig


    def get_svc_acc(self, gathered_val_vis, gathered_val_label, trainer):
        
        # method = SVC(kernel='rbf', max_iter=900000)
        method = SVC(kernel='linear', max_iter=900000)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
        n_scores = cross_val_score(
            method, 
            StandardScaler().fit_transform(gathered_val_vis), 
            gathered_val_label, 
            scoring="accuracy", 
            cv=cv, 
            n_jobs=5,
            )
        acc_mean = np.mean(n_scores)      
        return acc_mean 

    def get_svc_acc_rbf(self, gathered_val_vis, gathered_val_label, trainer):
        
        method = SVC(kernel='rbf', max_iter=900000)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
        n_scores = cross_val_score(
            method, 
            StandardScaler().fit_transform(gathered_val_vis), 
            gathered_val_label, 
            scoring="accuracy", 
            cv=cv, 
            n_jobs=5,
            )
        acc_mean = np.mean(n_scores)      
        return acc_mean 
    
    def on_validation_epoch_end(self, trainer, pl_module):
        
        num_val = len(self.val_vis)
        
        for val_index in range(num_val):
            
            val_name = f"val{val_index+1}"
            
            val_input_current = self.val_input[val_name]
            val_vis_current = self.val_vis[val_name]
            # val_vis_exp_current = self.val_vis_exp[val_name]
            hight_vis_current = self.val_high[val_name]
            val_label_current = self.val_label[val_name]
            
            print(val_name, len(val_vis_current), len(val_label_current))
            
            if trainer.max_epochs == 1 and pl_module.visited or len(val_vis_current) > 0 and trainer.current_epoch > 1 and (trainer.current_epoch+1) % self.inter == 0:
                
                if not isinstance(val_vis_current[0], np.ndarray):

                    val_input = torch.cat(val_input_current).cuda()
                    val_vis = torch.cat(val_vis_current).cuda()
                    # val_vis_exp = torch.cat(val_vis_exp_current).cuda()
                    hight_vis = torch.cat(hight_vis_current).cuda()
                    val_label = torch.cat(val_label_current).cuda()

                    gathered_val_input = trainer.strategy.all_gather(val_input).cpu().detach().numpy()
                    gathered_val_vis = trainer.strategy.all_gather(val_vis).cpu().detach().numpy()
                    # gathered_val_vis_exp = trainer.strategy.all_gather(val_vis_exp).cpu().detach().numpy()
                    gathered_hight_vis = trainer.strategy.all_gather(hight_vis).cpu().detach().numpy()
                    gathered_val_label = trainer.strategy.all_gather(val_label).cpu().detach().numpy()
                else:

                    gathered_val_input = np.concatenate(val_input_current)
                    gathered_val_vis = np.concatenate(val_vis_current)
                    # gathered_val_vis_exp = np.concatenate(val_vis_exp_current)
                    gathered_hight_vis = np.concatenate(hight_vis_current)
                    val_label = torch.cat(val_label_current).cuda()
                    gathered_val_label = trainer.strategy.all_gather(val_label).cpu().detach().numpy()
                if len(gathered_val_vis.shape) == 3:

                    gathered_val_input = gathered_val_input.reshape(-1, *gathered_val_input.shape[2:])
                    gathered_val_vis = gathered_val_vis.reshape(-1, *gathered_val_vis.shape[2:])
                    gathered_hight_vis = gathered_hight_vis.reshape(-1, *gathered_hight_vis.shape[2:])
                    gathered_val_label = gathered_val_label.reshape(-1)

                if trainer.is_global_zero:          # 检查是否为 rank 0 的进程
                    # print('gathered_val_vis.shape', gathered_val_vis.shape)

                    fig_scatter = self.plot_scatter(gathered_val_vis, gathered_val_label, trainer)
                    fig_scatter_hyper = self.plot_scatter_hyper(gathered_val_vis, gathered_val_label, trainer)
                    
                    if gathered_val_vis.shape[0] > 10000:
                        random_index = np.random.choice(gathered_val_vis.shape[0], 10000, replace=False)

                        gathered_val_input = gathered_val_input[random_index]
                        gathered_val_vis = gathered_val_vis[random_index]
                        gathered_hight_vis = gathered_hight_vis[random_index]
                        gathered_val_label = gathered_val_label[random_index]
                    
                    # import pdb; pdb.set_trace()
                    if gathered_val_vis.shape[1] > 2:
                        tsne = TSNE(n_components=2, random_state=0)
                        gathered_val_vis = tsne.fit_transform(gathered_val_vis)
                    
                    if len(gathered_val_input.shape) > 2:
                        gathered_val_input = gathered_val_input.reshape(gathered_val_input.shape[0],-1)
                
                    exp_scatter_log_dict = {}
                    # for exp_index in range(gathered_val_vis_exp.shape[1]):
                    #     fig_scatter_exp = self.plot_scatter(
                    #         gathered_val_vis_exp[:,exp_index,:], 
                    #         gathered_val_label, trainer)
                    #     exp_scatter_log_dict[f'scatter exp{exp_index}'] = wandb.Image(fig_scatter_exp)
                
                    # import pdb; pdb.set_trace()
                    acc_mean = self.get_svc_acc(
                        gathered_val_vis, 
                        gathered_val_label, 
                        trainer
                        )
                    acc_mean_rbf = self.get_svc_acc_rbf(
                        gathered_val_vis, 
                        gathered_val_label, 
                        trainer
                        )
                    # fig_heatmap_weight = go.Figure(data=go.Heatmap(
                    #     z=self.validation_weight[0].reshape(28,28).detach().cpu().numpy()))
                    # fig_heatmap_weight.update_layout(
                    #     title='Heatmap of Validation Weight',
                    #     xaxis_title='X-axis',
                    #     yaxis_title='Y-axis'
                    # )
                   
                    dataset_name = ['train',"validation","test"]
                    
                    dict_log = {
                        dataset_name[val_index]+"_svc": acc_mean, 
                        dataset_name[val_index]+"_svc_rbf": acc_mean_rbf,
                        "epoch": trainer.current_epoch,
                        dataset_name[val_index]+"_scatter": wandb.Image(fig_scatter),
                        dataset_name[val_index]+"_fig_scatter_hyper": wandb.Image(fig_scatter_hyper),
                        # 'fig_heatmap_weight': fig_heatmap_weight
                        } 
                    if self.fully_eval:
                        for kk in [120]:
                            ecb_e_train = ecb.Eval(
                                input=gathered_val_input,
                                latent=gathered_val_vis, 
                                label=gathered_val_label,
                                k=kk
                                )
                            trust = ecb_e_train.E_trustworthiness()
                            continuity = ecb_e_train.E_continuity()
                            dict_log[dataset_name[val_index]+"_trust"+str(kk)] = trust
                            dict_log[dataset_name[val_index]+"_continuity"+str(kk)] = continuity
                        
                        fknn = np.mean(evaluation.faster_knn_eval_series(gathered_val_vis,gathered_val_label))
                        fct = evaluation.faster_centroid_triplet_eval(gathered_val_input,gathered_val_vis,gathered_val_label)
                        dict_log[dataset_name[val_index]+"_fknn"]=fknn
                        dict_log[dataset_name[val_index]+"_fct"]=fct
                    exp_scatter_log_dict.update(dict_log)

                    trainer.logger.log_metrics(exp_scatter_log_dict)
                    plt.close()
                                        
                    print('------------------------------------------------------------------')
                    print('val_index', val_index, "SVC_val", acc_mean, "epoch", trainer.current_epoch)
                    print('------------------------------------------------------------------')

                    if not os.path.exists(self.dirpath):
                        os.makedirs(self.dirpath, exist_ok=True)
                    
                    if acc_mean > self.best_acc:
                        self.best_acc = acc_mean
                        torch.save(pl_module.state_dict(), os.path.join(
                            self.dirpath, f"best_model_{self.dataset}_acc{acc_mean}.pth"))
                            
                # if trainer.current_epoch > 400 and acc_mean < 0.8:
                #     # stop the training
                #     trainer.should_stop = True
            
        self.val_input = {'val1': [], 'val2': [], 'val3': []}    
        self.val_high = {'val1': [], 'val2': [], 'val3': []}
        self.val_vis = {'val1': [], 'val2': [], 'val3': []}
        self.val_label = {'val1': [], 'val2': [], 'val3': []}
        self.val_vis_exp = {'val1': [], 'val2': [], 'val3': []}
        
        

        

            
            