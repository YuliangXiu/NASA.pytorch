import sys
import math
import argparse
import cv2
import os
import numpy as np

import torch
import torchvision
import torch.nn.functional as F

from lib.common.trainer import Trainer
from lib.common.config import get_cfg_defaults
from lib.dataset.AMASSdataset import AMASSdataset
from lib.net.DeepSDF import Net
from lib.net.seg3d_lossless import Seg3dLossless

sys.path.insert(0, '../')

def find_vertices(sdf, direction="front"):
    '''
        - direction: "front" | "back" | "left" | "right"
    '''
    resolution = sdf.size(2)
    if direction == "front":
        pass
    elif direction == "left":
        sdf = sdf.permute(2, 1, 0)
    elif direction == "back":
        inv_idx = torch.arange(sdf.size(2)-1, -1, -1).long()
        sdf = sdf[inv_idx, :, :]
    elif direction == "right":
        inv_idx = torch.arange(sdf.size(2)-1, -1, -1).long()
        sdf = sdf[:, :, inv_idx]
        sdf = sdf.permute(2, 1, 0)

    inv_idx = torch.arange(sdf.size(2)-1, -1, -1).long()
    sdf = sdf[inv_idx, :, :]
    sdf_all = sdf.permute(2, 1, 0)

    # shadow
    grad_v = (sdf_all>0.5) * torch.linspace(resolution, 1, steps=resolution).to(sdf.device)
    grad_c = torch.ones_like(sdf_all) * torch.linspace(0, resolution-1, steps=resolution).to(sdf.device)
    max_v, max_c = grad_v.max(dim=2)
    shadow = grad_c > max_c.view(resolution, resolution, 1)
    keep = (sdf_all>0.5) & (~shadow)
    
    p1 = keep.nonzero().t() #[3, N]
    p2 = p1.clone() # z
    p2[2, :] = (p2[2, :]-2).clamp(0, resolution)
    p3 = p1.clone() # y
    p3[1, :] = (p3[1, :]-2).clamp(0, resolution)
    p4 = p1.clone() # x
    p4[0, :] = (p4[0, :]-2).clamp(0, resolution)

    v1 = sdf_all[p1[0, :], p1[1, :], p1[2, :]]
    v2 = sdf_all[p2[0, :], p2[1, :], p2[2, :]]
    v3 = sdf_all[p3[0, :], p3[1, :], p3[2, :]]
    v4 = sdf_all[p4[0, :], p4[1, :], p4[2, :]]

    X = p1[0, :].long() #[N,]
    Y = p1[1, :].long() #[N,]
    Z = p2[2, :].float() * (0.5 - v1) / (v2 - v1) + p1[2, :].float() * (v2 - 0.5) / (v2 - v1) #[N,]
    Z = Z.clamp(0, resolution)

    # normal
    norm_z = v2 - v1
    norm_y = v3 - v1
    norm_x = v4 - v1
    # print (v2.min(dim=0)[0], v2.max(dim=0)[0], v3.min(dim=0)[0], v3.max(dim=0)[0])

    norm = torch.stack([norm_x, norm_y, norm_z], dim=1)
    norm = norm / torch.norm(norm, p=2, dim=1, keepdim=True)

    return X, Y, Z, norm

def render_normal(resolution, X, Y, Z, norm):
    image =  torch.ones(
        (1, 3, resolution, resolution), dtype=torch.float32
    ).to(norm.device)  
    color = (norm + 1) / 2 
    color = color.clamp(0, 1)
    image[0, :, Y, X] = color.t()
    return image


class TestEngine():
    def __init__(self, query_func, device):
        self.device = device
        self.resolutions = [16+1, 32+1, 64+1, 128+1, 256+1]
        self.reconEngine = Seg3dLossless(
            query_func=query_func, 
            b_min=[[-1.0, -1.0, -1.0]],
            b_max=[[ 1.0,  1.0,  1.0]],
            resolutions=self.resolutions,
            balance_value=0.5,
            visualize=False,
            faster=True).to(device)

    def __call__(self, priors):

        with torch.no_grad():

            # forward
            sdf = self.reconEngine(priors=priors)[0,0]
            depth, height, width = sdf.size()
            # mask = F.interpolate(mask, size=(height, width))
            # sdf = sdf * (mask[0] > 0.1).float()

            # render
            X, Y, Z, norm = find_vertices(sdf, direction="front")
            image1 = render_normal(self.resolutions[-1], X, Y, Z, norm)
            X, Y, Z, norm = find_vertices(sdf, direction="left")
            image2 = render_normal(self.resolutions[-1], X, Y, Z, norm)
            X, Y, Z, norm = find_vertices(sdf, direction="right")
            image3 = render_normal(self.resolutions[-1], X, Y, Z, norm)
            X, Y, Z, norm = find_vertices(sdf, direction="back")
            image4 = render_normal(self.resolutions[-1], X, Y, Z, norm)

            image = torch.cat([image1, image2, image3, image4], axis=3)
            image = image.cpu().numpy()[0].transpose(1, 2, 0) * 255.0
            return np.uint8(image)  # rgb


if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-cfg', '--config_file', type=str, help='path of the yaml config file')
    argv = sys.argv[1:sys.argv.index('--')]
    args = parser.parse_args(argv)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    device = 'cuda:0'

    # set dataset
    test_dataset = AMASSdataset(cfg.dataset, split="test")
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1, shuffle=False,
        num_workers=12, pin_memory=True)

    bdata = next(iter(test_loader))

    # setup net 
    net = Net(21, 4, 40, 4).to(device)
    trainer = Trainer(net, cfg, use_tb=True)
    trainer.load_ckpt(
        os.path.join(trainer.checkpoints_path, 'ckpt_3.pth'))
    
    test_engine = TestEngine(trainer.query_func, device)
    render = test_engine(priors=bdata)

    cv2.imwrite(os.path.join(cfg.results_path, "render_result.jpg"), render[:, :, ::-1])


    