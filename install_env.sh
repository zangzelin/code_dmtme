# conda create -n nml python=3.9
conda activate nml
conda install pytorch torchvision torchaudio pytorch-cuda=12.3 -c pytorch -c nvidia
pip install plotly

pip install scikit-learn
pip install wandb 
pip install munkres
pip install matplotlib
pip install umap-learn
pip install pandas
pip install scanpy
# pip install pytorch_lightning==1.9
pip install lightning

pip install -U kaleido

pip install -U 'jsonargparse[signatures]>=4.26.1'

pip install einops
pip install transformers
conda install -c nvidia cuda-nvcc=12.3
pip install flash_attn
pip install -U deepspeed