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
save_dir = osp.join(opt['save_dir'], opt['exp_name'])

model_type = opt['model_type']


# get model according to yml config
if model_type == 'BaseSAREOModel':
    BaseModel = BaseSAREOModel
elif model_type == 'SemiSuperSAREOModel':
    BaseModel = SemiSuperSAREOModel
else:
    raise AttributeError('not valid model type')


model = BaseModel(opt)
model.inference_with_calibr()
# model.inference()

if os.path.exists("te.log"):
    os.system(f'cp te.log {save_dir}')
    print('\nSync tr.log to save_dir done')

print('[MAVOC] all done, everything ok')

