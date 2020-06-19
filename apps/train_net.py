# Usage: 
# python train_net.py -cfg ../configs/example.yaml -- learning_rate 1.0

import sys
import os
import argparse
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

sys.path.insert(0, '../')
from lib.common.trainer import Trainer
from lib.common.config import get_cfg_defaults
from lib.dataset.AMASSdataset import AMASSdataset
from lib.net.DeepSDF import Net

parser = argparse.ArgumentParser()
parser.add_argument(
    '-cfg', '--config_file', type=str, help='path of the yaml config file')
argv = sys.argv[1:sys.argv.index('--')]
args = parser.parse_args(argv)

# opts = sys.argv[sys.argv.index('--') + 1:]

# default cfg: defined in 'lib.common.config.py'
cfg = get_cfg_defaults()
cfg.merge_from_file(args.config_file)
# Now override from a list (opts could come from the command line)
# opts = ['dataset.root', '../data/XXXX', 'learning_rate', '1e-2']
# cfg.merge_from_list(opts)
cfg.freeze()



def test(net):
    net.eval()
    # set dataset
    test_dataset = AMASSdataset(cfg.dataset, split="test")
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=12, pin_memory=True)

    test_loss = 0
    correct = 0

    with torch.no_grad():
        pbar = tqdm(test_loader)
        for data_BX, data_BT, target in pbar:
            data_BX = data_BX.cuda()
            data_BT = data_BT.cuda()
            target = target.cuda()
            output = net(data_BX, data_BT)
            test_loss += F.mse_loss(output, target).item()

            pred = output.data
            pred = pred.masked_fill(pred<0.5, 0.)
            pred = pred.masked_fill(pred>=0.5, 1.)

            correct += pred.eq(target.data.view_as(pred)).float().mean()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        test_loss, 100. * correct / len(test_loader.dataset)))


def train(device='cuda'):
    # setup net 
    net = Net(21, 4, 40, 4).to(device)

    # setup trainer
    trainer = Trainer(net, cfg, use_tb=True)
    # load ckpt
    if os.path.exists(cfg.ckpt_path):
        trainer.load_ckpt(cfg.ckpt_path)
    else:
        trainer.logger.info(f'ckpt {cfg.ckpt_path} not found.')

    # set dataset
    train_dataset = AMASSdataset(cfg.dataset, split="train")

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_threads, pin_memory=True, drop_last=True)
    trainer.logger.info(
        f'train data size: {len(train_dataset)}; '+
        f'loader size: {len(train_data_loader)};')
    
    start_iter = trainer.iteration
    start_epoch = trainer.epoch
    # start training
    for epoch in range(start_epoch, cfg.num_epoch):
        trainer.net.train()

        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_threads, pin_memory=True, drop_last=True)
        loader = iter(train_data_loader)
        niter = len(train_data_loader)      
        
        epoch_start_time = iter_start_time = time.time()
        for iteration in range(start_iter, niter):

            # data_BX [B, N, 21, 3]
            # data_BT [B, N, 21, 3]
            # theta_out [B, N, 21]

            data_BX, data_BT, target = next(loader)  
               
            iter_data_time = time.time() - iter_start_time
            global_step = epoch * niter + iteration
            
            data_BX = data_BX.to(device)
            data_BT = data_BT.to(device)
            target = target.to(device)
            output = trainer.net(data_BX, data_BT)
            loss = F.mse_loss(output, target)

            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()

            iter_time = time.time() - iter_start_time
            eta = (niter-start_iter) * (time.time()-epoch_start_time) / (iteration-start_iter+1) 

            # print
            if iteration % cfg.freq_plot == 0 and iteration > 0:
                trainer.logger.info(
                    f'Name: {cfg.name}|Epoch: {epoch:02d}({iteration:05d}/{niter})|' \
                    +f'dataT: {(iter_data_time):.3f}|' \
                    +f'totalT: {(iter_time):.3f}|'
                    +f'ETA: {int(eta // 60):02d}:{int(eta - 60 * (eta // 60)):02d}|' \
                    +f'Err:{loss.item():.5f}|'
                )
                trainer.tb_writer.add_scalar('data/loss', loss.item(), global_step)

            # save
            if iteration % cfg.freq_save == 0 and iteration > 0:
                trainer.update_ckpt(
                    f'ckpt_{epoch}.pth', epoch, iteration)

            # evaluation
            if iteration % cfg.freq_eval == 0 and iteration > 0:
                trainer.net.eval()
                test(trainer.net.module)
                trainer.net.train()

            # end
            iter_start_time = time.time()
        
        trainer.scheduler.step()
        start_iter = 0


if __name__ == '__main__':
    train()