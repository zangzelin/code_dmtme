from data_model.datamodel_tri_test import MyDataModule
# from model.trainsfomer import DMTEVT_model
# from model.nnet import DMTEVT_model
from lightning import LightningModule
from lightning import LightningDataModule

from lightning.pytorch.cli import LightningCLI
import torch
torch.set_float32_matmul_precision('medium')
import wandb


# wandb.login(
#     host=,
#     key=,
# )

# only use one cpu
# torch.set_num_threads(1)

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.init_args.num_positive_samples", "model.init_args.num_positive_samples")
        parser.link_arguments("data.init_args.data_name", "model.init_args.data_name")
        parser.link_arguments("trainer.max_epochs", "model.init_args.max_epochs")


cli = MyLightningCLI(
    LightningModule,
    LightningDataModule,
    save_config_callback=None,
    subclass_mode_model=True, 
    subclass_mode_data=True,
    )

wandb.finish()