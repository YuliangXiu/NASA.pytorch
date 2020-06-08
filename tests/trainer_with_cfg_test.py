import sys
import os
import torchvision

sys.path.insert(0, '../')
from lib.common.trainer import Trainer
from lib.common.config import get_cfg_defaults

# default cfg: defined in 'lib.common.config.py'
cfg = get_cfg_defaults()
cfg.merge_from_file('../configs/example.yaml')

# Now override from a list (opts could come from the command line)
opts = ['dataset.root', '../data/XXXX', 'learning_rate', '1e-2']
cfg.merge_from_list(opts)

net = torchvision.models.AlexNet().to('cuda')

trainer = Trainer(net, cfg, use_tb=True)

trainer.update_ckpt(
    'test_ckpt.pth', epoch=0, iteration=0, score=0.0, loss=0.0)
trainer.load_ckpt(
    os.path.join(trainer.checkpoints_path, 'test_ckpt.pth'))