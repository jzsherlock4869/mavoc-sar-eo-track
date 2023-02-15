import sys
sys.path.append('../')
sys.path.append('./')

import yaml
import addict
import numpy as np
import os
import os.path as osp
from datetime import datetime
from glob import glob
import argparse

import torch
import random
torch.autograd.set_detect_anomaly(True)

import warnings
warnings.filterwarnings('ignore')

from models.base_sareo_model import BaseSAREOModel
from models.semisuper_sareo_model import SemiSuperSAREOModel

# =============================== #
#  seed all for re-implementation #
# =============================== #
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

# =============================== #
#  seed all for re-implementation #
# =============================== #

parser = argparse.ArgumentParser(description='input the configure file path')
parser.add_argument('--opt', type=str, required=True, help='config file path')
args = parser.parse_args()
config_path = args.opt

# load configs
with open(config_path, 'r') as f:
    opt = yaml.load(f, Loader=yaml.FullLoader)
opt = addict.Dict(opt)
# modify params in config on the fly
opt['exp_name'] = os.path.basename(config_path)[:-4]

# make dirs for save
save_dir = osp.join(opt['save_dir'], opt['exp_name'])
if os.path.exists(save_dir):
    timestamp = datetime.now().strftime("%Y%h%d%H%M")
    os.system(f'mv {save_dir} {save_dir}_archived_{timestamp}')
os.makedirs(save_dir, exist_ok=False)
os.system(f'cp {config_path} {save_dir}')
log_dir = osp.join(opt['log_dir'], opt['exp_name'])
if os.path.exists(log_dir):
    os.system(f'mv {log_dir} {log_dir}_archived_{timestamp}')
os.makedirs(log_dir, exist_ok=False)

eval_interval = opt['eval']['eval_interval']
num_epoch = opt['train']['num_epoch']
model_type = opt['model_type']

# get model according to yml config
if model_type == 'BaseSAREOModel':
    BaseModel = BaseSAREOModel
elif model_type == 'SemiSuperSAREOModel':
    BaseModel = SemiSuperSAREOModel
else:
    raise AttributeError('not valid model type')

# prepare for training 
model = BaseModel(opt)
model.prepare_training()

# training process
for epoch_id in range(num_epoch):
    print(f"\n[Train] Epoch {epoch_id}/{num_epoch}")
    model.train_epoch(epoch_id)
    if epoch_id % eval_interval == 0 or epoch_id > num_epoch - 10:
        print(f"\n[Eval] Epoch {epoch_id}/{num_epoch}")
        model.eval_epoch(epoch_id)
    if os.path.exists(f"logs/{opt['exp_name'][:3]}*.log"):
        os.system(f"cp logs/{opt['exp_name'][:3]}*.log {save_dir}")
        print(f"\nSync trainning logs to save_dir done")

print('[MAVOC] all done, everything ok')