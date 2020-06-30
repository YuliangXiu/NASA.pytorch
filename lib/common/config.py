from yacs.config import CfgNode as CN


_C = CN()

# needed by trainer
_C.name = 'default'
_C.checkpoints_path = '../data/checkpoints/'
_C.results_path = '../data/results/'
_C.learning_rate = 1e-3
_C.weight_decay = 0.0
_C.momentum = 0.0
_C.optim = 'RMSprop'
_C.schedule = [40, 60]
_C.gamma = 0.1
_C.resume = False 
_C.overfit = 0
_C.test_mode = False
_C.fast_dev = False

# needed by train()
_C.ckpt_path = ''
_C.batch_size = 4
_C.num_threads = 4
_C.num_epoch = 1
_C.freq_plot = 10
_C.freq_save = 100
_C.freq_eval = 100
_C.freq_show = 100

_C.net = CN()
_C.net.backbone = ''

_C.dataset = CN()
_C.dataset.root = ''
_C.dataset.num_sample_geo = 5000
_C.dataset.sigma_geo = 0.05
_C.dataset.num_verts = 2048
_C.dataset.sk_ratio = 1.0

_C.dataset.train_bsize = 1.0
_C.dataset.val_bsize = 1.0
_C.dataset.test_bsize = 1.0


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`

# cfg = get_cfg_defaults()
# cfg.merge_from_file('../configs/example.yaml')

# # Now override from a list (opts could come from the command line)
# opts = ['dataset.root', '../data/XXXX', 'learning_rate', '1e-2']
# cfg.merge_from_list(opts)