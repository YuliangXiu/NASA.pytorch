import sys
import os
import os.path as osp
import argparse
from tqdm import tqdm
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
from lib.dataset.mesh_util import calculate_fscore, calculate_mIoU, calculate_chamfer
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
        self.model_kwargs = kwargs
        self.global_step = 0
        self.batch_size = cfg.batch_size
        self.static_test_batch = None
        self.images = []
        self.testor_train = None
        self.testor_test = None
        self.tmux_logger = colorlogger(logdir=osp.join(cfg.results_path, cfg.name)) 

        self.hparams = {'lr': cfg.learning_rate,
                        'epoch': cfg.num_epoch,
                        'optim': cfg.optim,
                        'bsize': cfg.batch_size,
                        'sk_ratio': cfg.dataset.sk_ratio
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
            batch_size=1, shuffle=False,
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
            self.testor_train = TestEngine(self.query_func, self.device, False)
            self.static_test_batch = batch
            self.logger.experiment.add_graph(self.model, 
                                    (batch['data_BX'], batch['data_BT']))

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

        # for better acc
        output_max = output_max[:, :-(cfg.dataset.num_verts)]
        target = target[:, :-(cfg.dataset.num_verts)]

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
                        'p0': acc*100.0}


            if batch_idx % cfg.freq_show == 0:

                # render, verts, faces, colrs = self.testor_test(priors=self.static_test_batch)
                render = self.testor_train(priors=self.static_test_batch)[0,0]
                self.images.append(np.flip(render[:, :, ::-1],axis=0))
                imageio.mimsave(os.path.join(cfg.results_path, 
                                            cfg.name, 
                                            "results.gif"), self.images)
                self.logger.experiment.add_image(
                        tag=f'Image/{self.global_step}', 
                        img_tensor=np.flip(render[:, :, ::-1],axis=0).transpose(2,0,1), 
                        global_step=self.global_step)

                # self.logger.experiment.add_mesh(
                #         tag=f'mesh/{self.global_step}',
                #         vertices=verts.unsqueeze(0),
                #         faces=faces.unsqueeze(0),
                #         colors=(colrs.unsqueeze(0)*255).int(),
                #         global_step=self.global_step)

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

        # for better acc
        output_max = output_max[:, :-(cfg.dataset.num_verts)]
        target = target[:, :-(cfg.dataset.num_verts)]

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
        
        # self.logger.log_hyperparams(params=self.hparams, metrics=logs)

        return {'avg_val_loss': avg_loss, 
                'log': logs, 
                'progress_bar': logs_bar}

    def test_step(self, batch, batch_idx):

        self.testor_test = TestEngine(self.query_func, self.device, True)
        
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

        # for better acc
        output_max = output_max[:, :-(cfg.dataset.num_verts)]
        target = target[:, :-(cfg.dataset.num_verts)]

        output_max = output_max.masked_fill(output_max<0.5, 0.)
        output_max = output_max.masked_fill(output_max>0.5, 1.)

        acc = output_max.eq(target).float().mean()

        verts_gt = batch['verts'][0] # [B, 6890, 3]
        faces_gt = batch['faces'][0] # [B, 13776, 3]

        _, verts_pr, faces_pr, colrs_pr = self.testor_test(priors=batch)
        
        verts_pr -= 128.0
        verts_pr /= 128.0
        
        fscore, precision, recall = calculate_fscore(verts_gt, verts_pr)
        mIoU = calculate_mIoU(output_max, target)
        chamfer, p2s = calculate_chamfer(verts_gt, faces_gt, verts_pr, faces_pr)

        if batch_idx % 2 == 0:
            self.logger.experiment.add_mesh(
                            tag=f'mesh_pred/{batch_idx}',
                            vertices=verts_pr.unsqueeze(0),
                            faces=faces_pr.unsqueeze(0),
                            colors=(colrs_pr.unsqueeze(0)*255).int())

            self.logger.experiment.add_mesh(
                            tag=f'mesh_gt/{batch_idx}',
                            vertices=verts_gt.unsqueeze(0),
                            faces=faces_gt.unsqueeze(0))
        
        return {'test_loss': loss, 
                'test_acc': acc,
                'test_fscore': fscore,
                'test_mIoU': mIoU,
                'test_chamfer': chamfer}


    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        avg_fscore= np.array([x['test_fscore'] for x in outputs]).mean()
        avg_mIoU= np.array([x['test_mIoU'] for x in outputs]).mean()
        avg_chamfer= np.array([x['test_chamfer'] for x in outputs]).mean()
        
        metrics = {'test/loss': avg_loss.item(), 
                'test/acc': avg_acc.item(),
                'test/fscore': avg_fscore,
                'test/mIoU': avg_mIoU,
                'test/chamfer': avg_chamfer}

        self.logger.log_hyperparams(params=self.hparams, metrics=metrics)
        self.logger.save()

        return {'log': metrics} 


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
        save_top_k=5,
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

    trainer_kwargs = {
        'gpus':cfg.gpus,
        'logger':tb_logger,
        'callbacks':[LogCallback()],
        'checkpoint_callback':checkpoint,
        'progress_bar_refresh_rate':1,
        'limit_train_batches':cfg.dataset.train_bsize,
        'limit_val_batches':cfg.dataset.val_bsize,
        'limit_test_batches':cfg.dataset.test_bsize,
        'profiler':True,
        'num_sanity_val_steps':0,
        'fast_dev_run':cfg.fast_dev,
        'max_epochs':cfg.num_epoch,
        'val_check_interval':cfg.freq_eval,
        'row_log_interval':cfg.freq_plot,
        'log_save_interval':cfg.freq_save
    }

    model_kwargs = {
        'n_elements':21,
        'n_layers':4,
        'width':40,
        'D_dim':4
    }

    model = LightNASA(**model_kwargs)
    trainer = pl.Trainer(**trainer_kwargs)

    if cfg.overfit > 0:
        trainer_kwargs['overfit_batches'] = cfg.overfit
        trainer = pl.Trainer(**trainer_kwargs)
        
    if cfg.resume and osp.exists(cfg.ckpt_path):
        trainer_kwargs['resume_from_checkpoint'] = cfg.ckpt_path
        trainer = pl.Trainer(**trainer_kwargs)
        model.tmux_logger.info(f'Loading weights+params from {cfg.ckpt_path}.')
    elif not cfg.resume and osp.exists(cfg.ckpt_path):
        model.load_state_dict(torch.load(cfg.ckpt_path)['state_dict'])
        model.tmux_logger.info(f'Loading only weights from {cfg.ckpt_path}.')
    else:
        model.tmux_logger.info(f'ckpt {cfg.ckpt_path} not found.')

    if not cfg.test_mode:
        trainer.fit(model)
    
    np.random.seed(1993)
    trainer.test(model)
