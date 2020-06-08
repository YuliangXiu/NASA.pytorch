import sys
import os
import torchvision

sys.path.insert(0, '../')
from lib.common.trainer import Trainer

opt = Trainer.get_default_opt()
net = torchvision.models.AlexNet().to('cuda')

trainer = Trainer(net, opt, use_tb=True)

trainer.update_ckpt(
    'test_ckpt.pth', epoch=0, iteration=0, score=0.0, loss=0.0)
trainer.load_ckpt(
    os.path.join(trainer.checkpoints_path, 'test_ckpt.pth'))