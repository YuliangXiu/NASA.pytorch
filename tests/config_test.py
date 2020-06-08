import sys

sys.path.insert(0, '../')
from lib.common.config import get_cfg_defaults

cfg = get_cfg_defaults()
cfg.merge_from_file('../configs/example.yaml')

# Now override from a list (opts could come from the command line)
opts = ['dataset.root', '../data/XXXX', 'learning_rate', '1e-2']
cfg.merge_from_list(opts)

print (cfg)