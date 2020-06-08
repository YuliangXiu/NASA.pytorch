# Usage: 
# python train_net.py -cfg ../configs/example.yaml -- learning_rate 1.0

import sys
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

sys.path.insert(0, '../')
from lib.common.trainer import Trainer
from lib.common.config import get_cfg_defaults

parser = argparse.ArgumentParser()
parser.add_argument(
    '-cfg', '--config_file', type=str, help='path of the yaml config file')
argv = sys.argv[1:sys.argv.index('--')]
args = parser.parse_args(argv)

opts = sys.argv[sys.argv.index('--') + 1:]

# default cfg: defined in 'lib.common.config.py'
cfg = get_cfg_defaults()
cfg.merge_from_file(args.config_file)
# Now override from a list (opts could come from the command line)
# opts = ['dataset.root', '../data/XXXX', 'learning_rate', '1e-2']
cfg.merge_from_list(opts)
cfg.freeze()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def test(net):
    net.eval()
    # set dataset
    test_dataset = torchvision.datasets.MNIST(
        '../data/', train=False, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True)

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.cuda()
            target = target.cuda()
            output = net(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def train(device='cuda'):
    # -- TODO: change this line below --
    # setup net 
    net = Net().to(device)
    # ----

    # setup trainer
    trainer = Trainer(net, cfg, use_tb=True)
    # load ckpt
    if os.path.exists(cfg.ckpt_path):
        trainer.load_ckpt(cfg.ckpt_path)
    else:
        trainer.logger.info(f'ckpt {cfg.ckpt_path} not found.')

    # -- TODO: change this line below --
    # set dataset
    train_dataset = torchvision.datasets.MNIST(
        '../data/', train=True, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ]))
    # ----

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
            # -- TODO: change this line below --
            data, target = next(loader)         
            # ----
               
            iter_data_time = time.time() - iter_start_time
            global_step = epoch * niter + iteration
            
            # -- TODO: change this line below --
            data = data.to(device)
            target = target.to(device)
            output = trainer.net(data)
            loss = F.nll_loss(output, target)
            # ----

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
                # -- TODO: change this line below --
                test(trainer.net.module)
                # ----
                trainer.net.train()

            # end
            iter_start_time = time.time()
        
        trainer.scheduler.step()
        start_iter = 0


if __name__ == '__main__':
    train()