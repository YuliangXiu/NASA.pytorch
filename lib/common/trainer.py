import os
import sys
from easydict import EasyDict as edict

import torch
import torch.nn as nn

from tensorboardX import SummaryWriter
from .logger import colorlogger

class Trainer():
    def __init__(self, net, opt=None, use_tb=True):
        self.opt = opt if opt is not None else Trainer.get_default_opt()
        self.net = nn.DataParallel(net)
        self.net.train()

        # set cache path
        self.checkpoints_path = os.path.join(opt.checkpoints_path, opt.name)
        os.makedirs(self.checkpoints_path, exist_ok=True)
        self.results_path = os.path.join(opt.results_path, opt.name)
        os.makedirs(self.results_path, exist_ok=True)
        
        # set logger
        self.logger = colorlogger(logdir=self.results_path)
        self.logger.info(self.opt)
        
        # set tensorboard
        if use_tb:
            self.tb_writer = SummaryWriter(logdir=self.results_path)

        # set optimizer
        learning_rate = opt.learning_rate
        weight_decay = opt.weight_decay
        momentum = opt.momentum
        if opt.optim == "Adadelta":
            self.optimizer = torch.optim.Adadelta(
                self.net.parameters(), lr=learning_rate, 
                weight_decay=weight_decay)
        elif opt.optim == "SGD":
            self.optimizer = torch.optim.SGD(
                self.net.parameters(), lr=learning_rate, 
                momentum=momentum, weight_decay=weight_decay)
        elif opt.optim == "Adam":
            self.optimizer = torch.optim.Adam(
                self.net.parameters(), lr=learning_rate)
        elif opt.optim == "RMSprop":
            self.optimizer = torch.optim.RMSprop(
                self.net.parameters(), lr=learning_rate, 
                weight_decay=weight_decay, momentum=momentum)
        else:
            raise NotImplementedError
        
        # set scheduler
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=opt.schedule, gamma=opt.gamma)
        
        self.epoch = 0
        self.iteration = 0

    def update_ckpt(self, filename, epoch, iteration, **kwargs):
        # `kwargs` can be used to store loss, accuracy, epoch, iteration and so on.
        ckpt_path = os.path.join(self.checkpoints_path, filename)
        saved_dict = {
            "opt": self.opt,
            "net": self.net.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
            "iteration": iteration,
        }
        for k, v in kwargs.items():
            saved_dict[k] = v
        torch.save(saved_dict, ckpt_path)
        self.logger.info(f'save ckpt to {ckpt_path}')

    def load_ckpt(self, ckpt_path):
        self.logger.info(f'load ckpt from {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.net.module.load_state_dict(ckpt["net"])

        if self.opt.resume:
            self.logger.info('loading for optimizer & scheduler ...')
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            
            self.epoch = ckpt["epoch"]
            self.logger.info(f'loading for start epoch ... {self.epoch}')
            self.iteration = ckpt["iteration"]
            self.logger.info(f'loading for start iteration ... {self.iteration}')

    def query_func(self, priors, points):

        # points [B, N, 3]
        # priors['B_inv'] [B, 21, 4, 4]
        # priors['T_0'] [B, 4]
        
        B_inv = priors['B_inv'].cuda()
        T_0 = priors['T_0'].cuda()

        B, N, _ = points.shape

        # only recon one mesh
        if B_inv.shape[0] > 1:
            B_inv = B_inv[0].unsqueeze(0)
            T_0 = T_0[0].unsqueeze(0)

        # sample points [B, N, 1, 4, 1]
        X = torch.cat((points, 
            torch.ones(B, N, 1).cuda()),dim=2)[:, :, None, :, None] 

        # [B, 1, 21, 4, 4] x [B, N, 1, 4, 1] - > [B, N, 21, 4, 1] -- > [B, N, 21, 3]
        data_BX = torch.matmul(B_inv.unsqueeze(1), X)[:, :, :, :3, 0] 

        # [B, 1, 21, 4, 4] x [B, N, 1, 4, 1] - > [B, N, 21, 4, 1] -- > [B, N, 21, 3]
        data_BT = torch.matmul(B_inv.unsqueeze(1), T_0[:, None, None, :, None].repeat(
                                                    1, N, 1, 1, 1))[:, :, :, :3, 0]    

        # data_BX [B, N, 21, 3]
        # data_BT [B, N, 21, 3]

        pred = self.net(data_BX, data_BT)
        pred_max = torch.max(pred, dim=2)[0][:,None,:]
        # pred = self.test_recon(points)

        # return [B, 1, N]
        return pred_max

    def test_recon(self, points):

        from ..dataset.hoppeMesh import HoppeMesh
        import trimesh
        import numpy as np

        mesh_ori = trimesh.load_mesh("/home/ICT2000/yxiu/Code/trainer.pytorch/mesh.obj")

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
            points.detach().cpu().numpy()[0])).cuda()[None,None,:]
        
        return pred


    @classmethod
    def get_default_opt(cls):
        opt = edict()

        opt.name = 'example'
        opt.checkpoints_path = '../data/checkpoints/'
        opt.results_path = '../data/results/'
        opt.learning_rate = 1e-3
        opt.weight_decay = 0.0
        opt.momentum = 0.0
        opt.optim = 'RMSprop'
        opt.schedule = [40, 60]
        opt.gamma = 0.1
        opt.resume = False 
        return opt