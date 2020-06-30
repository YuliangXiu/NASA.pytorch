import sys
import os
import gc
import os.path as osp
import argparse
from tqdm import tqdm
import shutil
import torch
import trimesh
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
from lib.dataset.hoppeMesh import HoppeMesh
from lib.net.NASANet import NASANet
from lib.net.test_net import TestEngine

# pytorch lightning related libs
import pytorch_lightning as pl
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint

cfg = None


class LightNASA(pl.LightningModule):

    def __init__(self, **kwargs):
        super(LightNASA, self).__init__()
        self.model = NASANet(**kwargs)
        self.model_kwargs = kwargs
        self.global_step = 0
        self.batch_size = cfg.batch_size
        self.static_test_batch = None
        self.images = []
        self.testor = None
        self.trial = None
        self.tmux_logger = colorlogger(logdir=osp.join(cfg.results_path, cfg.name)) 
        
        if 'trial' in self.model_kwargs.keys():
            self.trial = self.model_kwargs['trial']

        self.hparams = {'lr': self.trial.suggest_float('lr', 
                            1e-6, 1e-2, log=True) if self.trial is not None \
                            else cfg.learning_rate,
                        'epoch': cfg.num_epoch,
                        'optim': self.trial.suggest_categorical('optimizer', 
                            ['SGD', 'Adam', 'RMSprop']) if self.trial is not None \
                            else cfg.optim,
                        'bsize': cfg.batch_size,
                        'sk_ratio': self.trial.suggest_float('sk_ratio',
                            0.1, 5.1, step=0.5) if self.trial is not None \
                            else cfg.dataset.sk_ratio
                    }
        self.hparams.update(self.model_kwargs)

    
    def forward(self, data_bx, data_bt):
        return self.model.forward(data_bx, data_bt)

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict

    def query_func(self, priors, points):
    
        B_inv = priors['B_inv'].to(self.device)
        T_0 = priors['T_0'].to(self.device)

        B, N, _ = points.shape

        if B_inv.shape[0] > 1:
            B_inv = B_inv[0].unsqueeze(0)
            T_0 = T_0[0].unsqueeze(0)

        X = torch.cat((points, 
            torch.ones(B, N, 1).to(self.device)),dim=2)[:, :, None, :, None] 

        data_BX = torch.matmul(B_inv.unsqueeze(1), X)[:, :, :, :3, 0] 
        data_BT = torch.matmul(B_inv.unsqueeze(1), T_0[:, None, None, :, None].repeat(
                                                    1, N, 1, 1, 1))[:, :, :, :3, 0]    
        pred = self.forward(data_BX, data_BT)
        pred_max = torch.max(pred, dim=2)[0][:,None,:]

        # fake reconstruction from original mesh
        # pred_max = self.fake_recon(points)

        return pred_max

    def fake_recon(self, points):

        mesh_ori = trimesh.load_mesh(osp.join(osp.dirname(__file__), "../data/mesh.obj"))

        rot_mat = trimesh.transformations.rotation_matrix(np.pi, [0,0,1])
        mesh_ori = mesh_ori.apply_transform(rot_mat)
        
        verts = mesh_ori.vertices
        vert_normals = mesh_ori.vertex_normals
        face_normals = mesh_ori.face_normals
        faces = mesh_ori.faces

        mesh = HoppeMesh(
            verts=verts, 
            faces=faces, 
            vert_normals=vert_normals, 
            face_normals=face_normals)
        pred = torch.Tensor(mesh.contains(
            points.detach().cpu().numpy()[0])).to(self.device)[None,None,:]
        
        return pred

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

        # self.tmux_logger.info(
        #     f'train data size: {len(self.train_dataset)}; '+
        #     f'loader size: {len(train_data_loader)};')
        
        return train_data_loader
    
    def val_dataloader(self):

        val_data_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size, shuffle=False,
            num_workers=cfg.num_threads, pin_memory=True)

        # self.tmux_logger.info(
        #     f'val data size: {len(self.val_dataset)}; '+
        #     f'loader size: {len(val_data_loader)};')
        
        return val_data_loader
    
    # Training related

    def configure_optimizers(self):

        # set optimizer
        learning_rate = self.hparams['lr']
        optim = self.hparams['optim']
        weight_decay = cfg.weight_decay
        momentum = cfg.momentum
        if optim == "Adadelta":
            self.optimizer = torch.optim.Adadelta(
                self.model.parameters(), lr=learning_rate, 
                weight_decay=weight_decay)
        elif optim == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=learning_rate, 
                momentum=momentum, weight_decay=weight_decay)
        elif optim == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=learning_rate)
        elif optim == "RMSprop":
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
            self.testor = TestEngine(self.query_func, self.device)
            self.static_test_batch = batch
         
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

        loss = loss_sample + self.hparams['sk_ratio'] * loss_skw

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
                        'p0': acc*100.0}

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

        loss = loss_sample + self.hparams['sk_ratio'] * loss_skw

        output_max = output_max.masked_fill(output_max<0.5, 0.)
        output_max = output_max.masked_fill(output_max>0.5, 1.)

        acc = output_max.eq(target).float().mean()

        output = {'val_loss':loss, 
                'val_acc':acc}

        return output

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        
        logs = {'val/loss': avg_loss, 
                'val/acc': avg_acc}

        logs_bar = {'l_val': avg_loss, 
                'p1': avg_acc*100.0}

        if self.trial is not None:
            self.trial.report(avg_acc, step=self.global_step)
            if self.trial.should_prune():
                raise optuna.TrialPruned()
        
        return {'avg_val_loss': avg_loss, 
                'log': logs, 
                'progress_bar': logs_bar}

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)

def objective(trial):

    # Filenames for each trial must be made unique in order to access each checkpoint.
    checkpoint = ModelCheckpoint(
        filepath=osp.join(cfg.checkpoints_path, cfg.name, 
                            "trial_{}".format(trial.number)),
        verbose=True,
        monitor='val/acc'
    )

    os.makedirs(osp.join(cfg.results_path, cfg.name), exist_ok=True)
    os.makedirs(osp.join(cfg.checkpoints_path, cfg.name), exist_ok=True)

    metrics_callback = MetricsCallback()

    trainer_kwargs = {
        'gpus':1,
        'logger':False,
        'distributed_backend':'dp',
        'checkpoint_callback':checkpoint,
        'progress_bar_refresh_rate':1,
        'limit_train_batches':cfg.dataset.train_bsize,
        'limit_val_batches':cfg.dataset.val_bsize,
        'num_sanity_val_steps':0,
        'max_epochs':cfg.num_epoch,
        'val_check_interval':cfg.freq_eval,
        'callbacks':[metrics_callback],
        'early_stop_callback':PyTorchLightningPruningCallback(trial, 
                                monitor="val/acc"),
    }

    model_kwargs = {
        'n_elements':21,
        'n_layers':trial.suggest_int("n_layers", 2, 12, step=2),
        'width':trial.suggest_int("width", 1*40, 21*40, step=40),
        'D_dim':trial.suggest_int("D_im", 2, 12, step=2),
        'trial':trial
    }

    model = LightNASA(**model_kwargs)
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model)


    return metrics_callback.metrics[-1]["val/acc"].item()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-cfg', '--config_file', type=str, help='path of the yaml config file')
    argv = sys.argv[1:sys.argv.index('--')]
    args = parser.parse_args(argv)
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, 
                    n_trials=20, 
                    timeout=600, 
                    n_jobs=1,
                    gc_after_trial=True, 
                    show_progress_bar=True)
    
    optuna.visualization.plot_intermediate_values(study)
    optuna.visualization.plot_optimization_history(study)
    optuna.visualization.plot_param_importances(study)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    shutil.rmtree(osp.join(cfg.results_path, cfg.name))
    shutil.rmtree(osp.join(cfg.checkpoints_path, cfg.name))
