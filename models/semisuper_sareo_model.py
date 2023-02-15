from distutils.command.build_scripts import first_line_re
import sys

sys.path.append('../')
import os
import os.path as osp

# torch related packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

# other packages
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import cv2

from data.sareo_dataloader import get_trainval_sareo_dataloader, get_test_sareo_dataloader, get_pseudo_sareo_dataloader
from archs.my_all_archs import SimpleDualstreamArch, ConcatInputArch
from utils.meter_utils import AverageMeter
from utils.net_utils import load_network
from losses.focal_loss import FocalLoss
from utils.calibr_utils import class_calibration, class_calibration_v2


"""
Semi supervised SAR+EO Model:
    input augmented sar and eo, output logits
    semi-supervised training strategy, from scratch, make the > thr test samples as trainset_v2
    re-train model with trainset and trainset_v2 continuously
    loss conducted on logits, focal or celoss
"""


class SemiSuperSAREOModel:
    """
    simple train, eval and test
    """
    def __init__(self, opt):
        self.opt = opt
        self.opt_dataset = opt['datasets']
        self.opt_train = opt['train']

        self.scale = opt['scale']
        self.device = opt['device']

        self.mean = [0, 0, 0]
        self.std = [1.0, 1.0, 1.0]

        self.num_classes = opt['train']['model_arch']['num_classes']
        self.max_acc = 0.0

    def prepare_training(self):

        os.makedirs(self.opt['log_dir'], exist_ok=True)
        log_path = osp.join(self.opt['log_dir'], self.opt['exp_name'])
        self.writer = SummaryWriter(log_path)

        # prepare network for training
        opt_model_arch = deepcopy(self.opt_train['model_arch'])
        arch_type = opt_model_arch.pop('type')
        load_path = opt_model_arch.pop('load_path') if 'load_path' in opt_model_arch else None
        self.network = self.get_network(arch_type, load_path, **opt_model_arch).to(self.device)
        if self.opt['multi_gpu']:
            self.network = nn.DataParallel(self.network)
        
        self.network.train()
        # prepare optimizer and corresponding net params
        opt_optim = deepcopy(self.opt_train['optimizer'])
        optim_type = opt_optim.pop('type')
        optim_params = []
        for k, v in self.network.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                print(f'Params {k} will not be optimized.')
        self.optimizer = self.get_optimizer(optim_type, optim_params, **opt_optim)

        # prepare lr scheduler
        opt_scheduler = deepcopy(self.opt_train['scheduler'])
        scheduler_type = opt_scheduler.pop('type')
        self.scheduler = self.get_scheduler(scheduler_type, self.optimizer, **opt_scheduler)

        # prepare dataloader
        self.train_loader, self.val_loader = self.get_trainval_dataloaders(self.opt_dataset, random_seed=42)

        # prepare criterion
        self.criterion = self.get_criterion(self.opt_train['criterion'])

    def get_criterion(self, opt_criterion):
        crit_type = opt_criterion.pop('type')
        if crit_type == 'CELoss':
            loss_func = nn.CrossEntropyLoss(**opt_criterion)
        elif crit_type == 'FocalLoss':
            loss_func = FocalLoss(**opt_criterion)
        else:
            raise NotImplementedError(f'loss func type {crit_type} is currently not supported')
        return loss_func

    # ============= get dataloaders ============== #

    def get_trainval_dataloaders(self, opt_dataset, random_seed=0):
        train_loader, val_loader = get_trainval_sareo_dataloader(opt_dataset, random_seed)
        return train_loader, val_loader

    def get_test_dataloaders(self, opt_dataset):
        test_loader = get_test_sareo_dataloader(opt_dataset)
        return test_loader

    def get_pseudo_dataloaders(self, opt_dataset):
        pseudo_loader = get_pseudo_sareo_dataloader(opt_dataset)
        return pseudo_loader

    # ============= get dataloaders ============== #

    def get_network(self, arch_type, load_path, **kwargs):
        if arch_type == 'SimpleDualstreamArch':
            network = SimpleDualstreamArch(**kwargs)
        elif arch_type == 'ConcatInputArch':
            network = ConcatInputArch(**kwargs)
        else:
            raise NotImplementedError(f'arch type {arch_type} is currently not supported')

        if load_path is not None:
            network = load_network(network, load_path)
        print(f"[MODEL] Load pretrained Generator from {load_path}")
        return network


    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        elif optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supperted yet.')
        return optimizer


    def get_scheduler(self, scheduler_type, optimizer, **kwargs):
        if scheduler_type == 'CosineAnnealingWarmRestarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **kwargs)
        elif scheduler_type == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
        elif scheduler_type == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
        elif scheduler_type == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
        else:
            raise NotImplementedError(f'scheduler type {scheduler_type} unrecoginized.')
        return scheduler


    def train_epoch(self, epoch_id):
        if epoch_id == 0:
            self.first_dive = self.opt_train['start_semi'][0]
            self.conf_dict = dict()
            for i in range(len(self.opt_train['start_semi'])):
                self.conf_dict[self.opt_train['start_semi'][i]] = self.opt_train['conf_thr'][i]
        if epoch_id < self.first_dive:
            self.train_epoch_stage1(epoch_id)
        else:
            if epoch_id in self.conf_dict:
                self.pseudo_labeling(conf_thr=self.conf_dict[epoch_id])
            self.train_epoch_stage2(epoch_id)
            

    def train_epoch_stage1(self, epoch_id):

        # re-shuffle and re-sample to make full use of majority class samples
        # self.train_loader, self.val_loader = self.get_trainval_dataloaders(self.opt_dataset, random_seed=epoch_id)

        self.network.train()
        batch_size = self.opt_dataset['trainval']['batch_size']
        loss_tot_avm = AverageMeter()

        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        # for iter_id, batch in enumerate(self.train_loader):
        for iter_id, batch in pbar:
            img_sar, img_eo = batch['img_sar'], batch['img_eo']

            img_sar, img_eo = img_sar.to(self.device), img_eo.to(self.device)
            labels = batch['class_id'].to(self.device)

            self.optimizer.zero_grad()

            out = self.network(img_sar, img_eo)
            loss_val = self.criterion(out, labels)

            loss_tot = loss_val

            loss_tot.backward()
            self.optimizer.step()

            loss_tot_avm.update(loss_tot.detach().item(), batch_size)
            # loss_consist_avm.update(l_consist.detach().item(), batch_size)
            # print(f"Train_Loss: {loss_score.avg}, Epoch: {epoch_id} iter {iter_id}, LR: {self.optimizer.param_groups[0]['lr']}")
            pbar.set_postfix(Loss_CE_Avg=loss_tot_avm.avg, Epoch=epoch_id, LR=self.optimizer.param_groups[0]['lr'])
            
            self.writer.add_scalar('loss/loss', loss_tot_avm.avg, epoch_id)
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch_id)
        
        self.scheduler.step()


    def train_epoch_stage2(self, epoch_id):

        # re-shuffle and re-sample to make full use of majority class samples
        # self.train_loader, self.val_loader = self.get_trainval_dataloaders(self.opt_dataset, random_seed=epoch_id)

        self.network.train()
        batch_size = self.opt_dataset['trainval']['batch_size']
        loss_tot_avm = AverageMeter()

        # train on real train dataset
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        # for iter_id, batch in enumerate(self.train_loader):
        for iter_id, batch in pbar:
            img_sar, img_eo = batch['img_sar'], batch['img_eo']

            img_sar, img_eo = img_sar.to(self.device), img_eo.to(self.device)
            labels = batch['class_id'].to(self.device)

            self.optimizer.zero_grad()

            out = self.network(img_sar, img_eo)
            loss_val = self.criterion(out, labels)

            loss_tot = loss_val

            loss_tot.backward()
            self.optimizer.step()

            loss_tot_avm.update(loss_tot.detach().item(), batch_size)
            # loss_consist_avm.update(l_consist.detach().item(), batch_size)
            # print(f"Train_Loss: {loss_score.avg}, Epoch: {epoch_id} iter {iter_id}, LR: {self.optimizer.param_groups[0]['lr']}")
            pbar.set_postfix(Loss_CE_Avg=loss_tot_avm.avg, TrainEpoch=epoch_id, LR=self.optimizer.param_groups[0]['lr'])
            
            self.writer.add_scalar('loss/loss', loss_tot_avm.avg, epoch_id)
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch_id)

        # train on pseudo labeled samples 
        if self.use_semi:
            print("[Train Stage 2] using semi supervised pseudo labeled dataset for training")
            pbar = tqdm(enumerate(self.pseudo_loader), total=len(self.pseudo_loader))
            for iter_id, batch in pbar:
                img_sar, img_eo = batch['img_sar'], batch['img_eo']

                img_sar, img_eo = img_sar.to(self.device), img_eo.to(self.device)
                labels = batch['class_id'].to(self.device)

                self.optimizer.zero_grad()

                out = self.network(img_sar, img_eo)
                loss_val = self.criterion(out, labels)

                loss_tot = loss_val

                loss_tot.backward()
                self.optimizer.step()

                loss_tot_avm.update(loss_tot.detach().item(), batch_size)
                # loss_consist_avm.update(l_consist.detach().item(), batch_size)
                # print(f"Train_Loss: {loss_score.avg}, Epoch: {epoch_id} iter {iter_id}, LR: {self.optimizer.param_groups[0]['lr']}")
                pbar.set_postfix(Loss_CE_Avg=loss_tot_avm.avg, PseudoEpoch=epoch_id, LR=self.optimizer.param_groups[0]['lr'])
                
                self.writer.add_scalar('loss/loss', loss_tot_avm.avg, epoch_id)
                self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch_id)
        else:
            print("[Train Stage 2] !!! NO PSUEDO LABELED DATA COLLECTED, PLEASE CHECK IF CONF_THR IS RESONABLE !!!")

        self.scheduler.step()


    def pseudo_labeling(self, conf_thr=0.95):

        self.test_loader = self.get_test_dataloaders(self.opt_dataset)
        # prepare network for training
        opt_model_arch = deepcopy(self.opt['test']['model_arch'])
        arch_type = opt_model_arch.pop('type')
        load_path = opt_model_arch.pop('load_path') if 'load_path' in opt_model_arch else None
        if load_path is None:
            load_path = osp.join(self.opt['save_dir'], self.opt['exp_name'], 'ckpt', 'best.pth.tar')
        self.network = self.get_network(arch_type, load_path, **opt_model_arch).to(self.device)
        self.network.eval()
        # fname = self.opt['test']['save_filename']
        os.makedirs(osp.join(self.opt['save_dir'], self.opt['exp_name'], 'pseudo_labels'), exist_ok=True)
        pseudo_info_csv = osp.join(self.opt['save_dir'], self.opt['exp_name'], 'pseudo_labels', f'test_pseudo_info.csv')
        if not osp.exists(pseudo_info_csv):
            fto = open(pseudo_info_csv, 'w')
            self.opt_dataset['pseudo']['info_csv'] = pseudo_info_csv
            fto.write("image_id,sar_path,eo_path,class_id,confidence")
        else:
            fto = open(pseudo_info_csv, 'a')

        num_each_class = dict([(str(i), 0) for i in range(self.num_classes)])
        pseudo_count = 0
        with torch.no_grad():
            pbar = tqdm(enumerate(self.test_loader), total=len(self.test_loader))
            # for iter_id, batch in enumerate(self.val_loader):
            for iter_id, batch in pbar:
                img_sar, img_eo = batch['img_sar'], batch['img_eo']
                img_sar, img_eo = img_sar.to(self.device), img_eo.to(self.device)
                imid = batch['imid'].item()

                out = self.network(img_sar, img_eo)
                # empirical weights to make output basically balanced, aborted.
                # out = out * torch.Tensor([0.1, 0.1, 0.1, 0.1, 0.6, 0.6, 0.9, 0.9, 1.9, 1.9]).to(self.device)
                probs = F.softmax(out, dim=1)
                max_prob = torch.max(probs).item()
                if max_prob > conf_thr:
                    pseudo_count += 1
                    predictions = out.argmax(dim=1, keepdim=True).squeeze().item()
                    num_each_class[str(predictions)] += 1
                    fto.write(f"\n{imid},test_images/test_sar/SAR_{imid}.png,test_images/test_eo/EO_{imid}.png,{predictions}, {max_prob}")
                pbar.set_postfix(Idx=iter_id, **num_each_class)

            print(f"\n\t >>> [Pseudo Summary] total {len(self.test_loader)} test images, pseudo labeled {pseudo_count} of them")
            for idx in num_each_class:
                print(f"\t ====== Class {idx} number: {num_each_class[idx]}")
        fto.close()

        if pseudo_count == 0:
            self.use_semi = False
        else:
            self.use_semi = True
            self.pseudo_loader = self.get_pseudo_dataloaders(self.opt_dataset)


    def eval_epoch(self, epoch_id):
        self.network.eval()
        print(f"[Eval] Begin validating set ")
        acc_avm = AverageMeter()
        acc_classwise_avm = [AverageMeter() for _ in range(self.num_classes)]
        confusion_mat = np.zeros((self.num_classes, self.num_classes))
        with torch.no_grad():
            pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
            # for iter_id, batch in enumerate(self.val_loader):
            for iter_id, batch in pbar:
                img_sar, img_eo = batch['img_sar'], batch['img_eo']
                img_sar, img_eo = img_sar.to(self.device), img_eo.to(self.device)
                labels = batch['class_id'].to(self.device)
                imid = batch['imid'].item()

                out = self.network(img_sar, img_eo)
                predictions = out.argmax(dim=1, keepdim=True).squeeze()
                # print("val iter ", iter_id, imid, predictions.item(), labels.item())
                if predictions.item() == labels.item():
                    acc_avm.update(1, 1)
                    acc_classwise_avm[labels.item()].update(1,1)

                else:
                    acc_avm.update(0, 1)
                    acc_classwise_avm[labels.item()].update(0,1)

                confusion_mat[predictions.item(), labels.item()] += 1
                # print(f"Train_Loss: {acc_avm.avg}, Epoch: {epoch_id} iter {iter_id}, LR: {self.optimizer.param_groups[0]['lr']}")
                pbar.set_postfix(Idx=iter_id, Acc_avg=acc_avm.avg)

            print(f"\n\t >>> [Eval Summary] Epoch {epoch_id}, total {len(self.val_loader)} test images, "
                    f"Overall Acc {acc_avm.avg}, mAcc {np.mean([acc_classwise_avm[idx].avg for idx in range(self.num_classes)])}")

            for idx in range(self.num_classes):
                print(f"\t ====== Class {idx} acc: {acc_classwise_avm[idx].avg}, "
                        f"correct {acc_classwise_avm[idx].sum} / total {acc_classwise_avm[idx].count}")

            # print("***************** CONFUSION MATRIX ********************")
            # print(confusion_mat)
            # print("***************** CONFUSION MATRIX ********************")

            self.writer.add_scalar(f'eval/overall_acc', acc_avm.avg, epoch_id)

            if acc_avm.avg > self.max_acc:
                self.save_model(epoch_id, acc_avm.avg, copy_best=True)
                self.max_acc = acc_avm.avg


    def save_model(self, epoch_id, val_metric, copy_best=True):
        save_dir = osp.join(self.opt['save_dir'], self.opt['exp_name'], 'ckpt')
        os.makedirs(save_dir, exist_ok=True)
        save_path = osp.join(save_dir, f'epoch{epoch_id:05}_metric{val_metric:.4f}.pth.tar')
        torch.save(self.network.state_dict(), save_path)
        if copy_best:
            best_path = osp.join(save_dir, 'best.pth.tar')
            torch.save(self.network.state_dict(), best_path)
        return save_path


    def inference(self):

        self.test_loader = self.get_test_dataloaders(self.opt_dataset)
        # prepare network for training
        opt_model_arch = deepcopy(self.opt['test']['model_arch'])
        arch_type = opt_model_arch.pop('type')
        load_path = opt_model_arch.pop('load_path') if 'load_path' in opt_model_arch else None
        if load_path is None:
            load_path = osp.join(self.opt['save_dir'], self.opt['exp_name'], 'ckpt', 'best.pth.tar')
        self.network = self.get_network(arch_type, load_path, **opt_model_arch).to(self.device)
        self.network.eval()

        fname = self.opt['test']['save_filename']

        os.makedirs(osp.join(self.opt['save_dir'], self.opt['exp_name'], 'prediction'), exist_ok=True)
        fto = open(osp.join(self.opt['save_dir'], self.opt['exp_name'], 'prediction', fname), 'w')
        fto.write("image_id,class_id\n")

        num_each_class = dict([(str(i), 0) for i in range(self.num_classes)])

        with torch.no_grad():
            pbar = tqdm(enumerate(self.test_loader), total=len(self.test_loader))
            # for iter_id, batch in enumerate(self.val_loader):
            for iter_id, batch in pbar:
                img_sar, img_eo = batch['img_sar'], batch['img_eo']
                img_sar, img_eo = img_sar.to(self.device), img_eo.to(self.device)
                imid = batch['imid'].item()

                out = self.network(img_sar, img_eo)
                # empirical weights to make output basically balanced, aborted.
                # out = out * torch.Tensor([0.1, 0.1, 0.1, 0.1, 0.6, 0.6, 0.9, 0.9, 1.9, 1.9]).to(self.device)
                predictions = out.argmax(dim=1, keepdim=True).squeeze()
                # print("val iter ", iter_id, imid, predictions.item(), labels.item())
                num_each_class[str(predictions.item())] += 1

                if iter_id < len(self.test_loader) - 1:
                    fto.write(f"{imid},{predictions.item()}\n")
                else:
                    fto.write(f"{imid},{predictions.item()}")
                # print(f"Train_Loss: {acc_avm.avg}, Epoch: {epoch_id} iter {iter_id}, LR: {self.optimizer.param_groups[0]['lr']}")
                pbar.set_postfix(Idx=iter_id, **num_each_class)

            print(f"\n\t >>> [Test Summary] total {len(self.test_loader)} test images")

            for idx in num_each_class:
                print(f"\t ====== Class {idx} number: {num_each_class[idx]}")

        fto.close()



    def inference_with_calibr(self):

        self.test_loader = self.get_test_dataloaders(self.opt_dataset)
        # prepare network for training
        opt_model_arch = deepcopy(self.opt['test']['model_arch'])
        arch_type = opt_model_arch.pop('type')
        load_path = opt_model_arch.pop('load_path') if 'load_path' in opt_model_arch else None
        if load_path is None:
            load_path = osp.join(self.opt['save_dir'], self.opt['exp_name'], 'ckpt', 'best.pth.tar')
        self.network = self.get_network(arch_type, load_path, **opt_model_arch).to(self.device)
        self.network.eval()

        fname = self.opt['test']['save_filename']

        os.makedirs(osp.join(self.opt['save_dir'], self.opt['exp_name'], 'prediction'), exist_ok=True)
        fto = open(osp.join(self.opt['save_dir'], self.opt['exp_name'], 'prediction', fname), 'w')
        fto.write("image_id,class_id\n")

        # num_each_class = dict([(str(i), 0) for i in range(self.num_classes)])

        imid_sort_ls = []
        with torch.no_grad():
            pbar = tqdm(enumerate(self.test_loader), total=len(self.test_loader))
            # for iter_id, batch in enumerate(self.val_loader):
            for iter_id, batch in pbar:
                img_sar, img_eo = batch['img_sar'], batch['img_eo']
                img_sar, img_eo = img_sar.to(self.device), img_eo.to(self.device)
                imid = batch['imid'].item()
                imid_sort_ls.append(imid)

                out = self.network(img_sar, img_eo)

                if iter_id == 0:
                    all_probs_test = F.softmax(out)
                else:
                    all_probs_test = torch.cat((all_probs_test, F.softmax(out)), dim=0)

            pbar.set_postfix(Idx=iter_id)

        # pred_class = class_calibration(all_probs_test)
        pred_class = class_calibration_v2(all_probs_test)        # (boost 27.36 -> 30.75, +3.39)

        for iter_id, batch in enumerate(self.test_loader):
            imid = batch['imid'].item()
            assert imid == imid_sort_ls[iter_id]
            predictions = pred_class[iter_id]

            if iter_id < len(self.test_loader) - 1:
                fto.write(f"{imid},{int(predictions.item())}\n")
            else:
                fto.write(f"{imid},{int(predictions.item())}")
            # print(f"Train_Loss: {acc_avm.avg}, Epoch: {epoch_id} iter {iter_id}, LR: {self.optimizer.param_groups[0]['lr']}")

        print(f"\n\t >>> [Test Summary] total {len(self.test_loader)} test images")

        print(torch.unique(pred_class, return_counts=True))

        fto.close()





















