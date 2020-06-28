import sys
import os
import os.path as osp
import argparse
from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import imageio

sys.path.insert(0, '../')
from lib.common.config import get_cfg_defaults
from lib.common.logger import colorlogger
from lib.dataset.AMASSdataset import AMASSdataset
from lib.net.NASANet import NASANet
from lib.net.test_net import TestEngine

# pytorch lightning related libs
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint


class LightNASA(pl.LightningModule):

    def __init__(self, **kwargs):
        super(LightNASA, self).__init__()
        self.model = NASANet(**kwargs)

        self.testor = TestEngine(self.query_func)
        self.global_step = 0
        self.batch_size = cfg.batch_size
        self.static_test_batch = None
        self.images = []
        self.tmux_logger = colorlogger(logdir=osp.join(cfg.results_path, cfg.name)) 

    def forward(self, data_bx, data_bt):
        return self.model.forward(data_bx, data_bt)

    def query_func(self, priors, points):
    
        B_inv = priors['B_inv'].cuda()
        T_0 = priors['T_0'].cuda()

        B, N, _ = points.shape

        if B_inv.shape[0] > 1:
            B_inv = B_inv[0].unsqueeze(0)
            T_0 = T_0[0].unsqueeze(0)

        X = torch.cat((points, 
            torch.ones(B, N, 1).cuda()),dim=2)[:, :, None, :, None] 

        data_BX = torch.matmul(B_inv.unsqueeze(1), X)[:, :, :, :3, 0] 
        data_BT = torch.matmul(B_inv.unsqueeze(1), T_0[:, None, None, :, None].repeat(
                                                    1, N, 1, 1, 1))[:, :, :, :3, 0]    
        pred = self.forward(data_BX, data_BT)
        pred_max = torch.max(pred, dim=2)[0][:,None,:]

        return pred_max

    # Dataset related

    def prepare_data(self):

        self.train_dataset = AMASSdataset(cfg, split="train")
        self.test_dataset = AMASSdataset(cfg, split="test")
        self.val_dataset = AMASSdataset(cfg, split="vald")

    def train_dataloader(self):

        train_data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size, shuffle=True,
            num_workers=cfg.num_threads, pin_memory=True)

        self.tmux_logger.info(
            f'train data size: {len(self.train_dataset)}; '+
            f'loader size: {len(train_data_loader)};')
        
        return train_data_loader
    
    def val_dataloader(self):

        val_data_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size, shuffle=False,
            num_workers=cfg.num_threads, pin_memory=True)

        self.tmux_logger.info(
            f'val data size: {len(self.val_dataset)}; '+
            f'loader size: {len(val_data_loader)};')
        
        return val_data_loader
    
    def test_dataloader(self):

        test_data_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size, shuffle=False,
            num_workers=cfg.num_threads, pin_memory=True)
        
        self.tmux_logger.info(
            f'test data size: {len(self.test_dataset)}; '+
            f'loader size: {len(test_data_loader)};')

        return test_data_loader

    # Training related

    def configure_optimizers(self):

        # set optimizer
        learning_rate = cfg.learning_rate
        weight_decay = cfg.weight_decay
        momentum = cfg.momentum
        if cfg.optim == "Adadelta":
            self.optimizer = torch.optim.Adadelta(
                self.model.parameters(), lr=learning_rate, 
                weight_decay=weight_decay)
        elif cfg.optim == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=learning_rate, 
                momentum=momentum, weight_decay=weight_decay)
        elif cfg.optim == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=learning_rate)
        elif cfg.optim == "RMSprop":
            self.optimizer = torch.optim.RMSprop(
                self.model.parameters(), lr=learning_rate, 
                weight_decay=weight_decay, momentum=momentum)
        else:
            raise NotImplementedError
        
        # set scheduler
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=cfg.schedule, gamma=cfg.gamma)

        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):

        if self.static_test_batch is None:
            self.static_test_batch = batch

        self.global_step = self.current_epoch * (len(self.train_dataset) // self.batch_size) + batch_idx

        data_BX, data_BT, target, weights = \
                batch['data_BX'], \
                batch['data_BT'], \
                batch['targets'], \
                batch['weights']
            
        output = self.forward(data_BX, data_BT)
        output_max = torch.max(output, dim=2)[0]
        output_verts = output[:, -(cfg.dataset.num_verts):, :]

        loss_sample = F.mse_loss(output_max, target)        
        loss_skw = F.mse_loss(output_verts, weights)

        loss = loss_sample + cfg.dataset.sk_ratio * loss_skw

        output_max = output_max.masked_fill(output_max<0.5, 0.)
        output_max = output_max.masked_fill(output_max>0.5, 1.)

        acc = output_max.eq(target).float().mean()

        logs = {}
        logs_bar = {}

        if batch_idx > 0:

            if batch_idx % cfg.freq_plot ==0:

                logs = {'train/loss_total': loss,
                        'train/loss_sample': loss_sample,
                        'train/loss_weights': loss_skw,
                        'train/accuracy': acc}

                logs_bar = {
                        'lr': self.scheduler.get_last_lr()[0],
                        'l_pc': loss_sample,
                        'l_sk': loss_skw,
                        'acc': acc}


            if batch_idx % cfg.freq_show == 0:

                render = self.testor(priors=self.static_test_batch)
                self.images.append(np.flip(render[:, :, ::-1],axis=0))
                imageio.mimsave(os.path.join(cfg.results_path, 
                                            cfg.name, 
                                            "results.gif"), self.images)
                self.logger.experiment.add_image('Image', 
                        np.flip(render[:, :, ::-1],axis=0).transpose(2,0,1), 
                        self.global_step)

        return {'loss': loss, 'log': logs, 'progress_bar': logs_bar}

    def validation_step(self, batch, batch_idx):
        
        data_BX, data_BT, target, weights = \
                batch['data_BX'], \
                batch['data_BT'], \
                batch['targets'], \
                batch['weights']
            
        output = self.forward(data_BX, data_BT)
        output_max = torch.max(output, dim=2)[0]
        output_verts = output[:, -(cfg.dataset.num_verts):, :]

        loss_sample = F.mse_loss(output_max, target)        
        loss_skw = F.mse_loss(output_verts, weights)

        loss = loss_sample + cfg.dataset.sk_ratio * loss_skw

        output_max = output_max.masked_fill(output_max<0.5, 0.)
        output_max = output_max.masked_fill(output_max>0.5, 1.)

        acc = output_max.eq(target).float().mean()

        output = {'val_loss':loss, 
                'val_acc':acc}

        return output


    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}] 
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        
        logs = {'val/loss': avg_loss, 
                'val/acc': avg_acc}

        return {'avg_val_loss': avg_loss, 
                'log': logs, 
                'progress_bar': logs}

    def test_step(self, batch, batch_idx):
        
        data_BX, data_BT, target, weights = \
                batch['data_BX'], \
                batch['data_BT'], \
                batch['targets'], \
                batch['weights']
            
        output = self.forward(data_BX, data_BT)
        output_max = torch.max(output, dim=2)[0]
        output_verts = output[:, -(cfg.dataset.num_verts):, :]

        loss_sample = F.mse_loss(output_max, target)        
        loss_skw = F.mse_loss(output_verts, weights)

        loss = loss_sample + cfg.dataset.sk_ratio * loss_skw

        output_max = output_max.masked_fill(output_max<0.5, 0.)
        output_max = output_max.masked_fill(output_max>0.5, 1.)

        acc = output_max.eq(target).float().mean()

        return {'test_loss': loss, 'test_acc': acc}


    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        logs = {'test/loss': avg_loss, 
                'test/acc': avg_acc}

        return {'avg_test_loss': avg_loss, 
                'log': logs, 
                'progress_bar': logs} 

    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-cfg', '--config_file', type=str, help='path of the yaml config file')
    argv = sys.argv[1:sys.argv.index('--')]
    args = parser.parse_args(argv)
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    
    os.makedirs(osp.join(cfg.results_path, cfg.name), exist_ok=True)
    os.makedirs(osp.join(cfg.checkpoints_path, cfg.name), exist_ok=True)

    osp.join(cfg.results_path, cfg.name)
    tb_logger = pl_loggers.TensorBoardLogger(cfg.results_path, name=cfg.name)

    checkpoint = ModelCheckpoint(
        filepath=osp.join(cfg.checkpoints_path, cfg.name),
        save_top_k=3,
        verbose=True,
        monitor='avg_val_loss',
        mode='min',
        prefix='top_k_'
    )

    class LogCallback(pl.Callback):
        def on_init_start(self, trainer):
            print('Starting to init trainer!')

        def on_init_end(self, trainer):
            print('trainer is init now')

        def on_train_start(self, trainer, pl_module):
            print("Training is started!\n")
            # print(f"Hyper Parameters: {cfg}\n")
        def on_train_end(self, trainer, pl_module):
            print("Training is done.\n")

    kwargs = {
        'gpus':1,
        'logger':tb_logger,
        'callbacks':[LogCallback()],
        'checkpoint_callback':checkpoint,
        'progress_bar_refresh_rate':1,
        'limit_train_batches':0.1,
        'limit_val_batches':0.1,
        'limit_test_batches':1.0,
        'profiler':True,
        'fast_dev_run':False,
        'max_epochs':cfg.num_epoch,
        'val_check_interval':cfg.freq_eval,
        'row_log_interval':cfg.freq_plot,
        'log_save_interval':cfg.freq_save
    }

    trainer = pl.Trainer(**kwargs)

    if cfg.overfit:
        kwargs['overfit_batches'] = 2
        trainer = pl.Trainer(**kwargs)

    if cfg.resume and osp.exists(cfg.ckpt_path):
        kwargs['resume_from_checkpoint'] = cfg.ckpt_path
        trainer = pl.Trainer(**kwargs)
    elif not cfg.resume and osp.exists(cfg.ckpt_path):
        trainer.load_from_checkpoint(cfg.ckpt_path)
    else:
        print(f'ckpt {cfg.ckpt_path} not found.')
    
    model = LightNASA(n_elements=21, n_layers=4, width=40, D_dim=4)
    trainer.fit(model)
    trainer.test()
