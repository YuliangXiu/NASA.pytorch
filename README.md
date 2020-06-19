# trainer.pytorch

A tiny but handy training code base for general tasks, written in pytorch. Features in this codes are:

- support tensorboardx by default.
- support logger & timing by default.
- support multi-gpu (nn.DataParallel) by default.
- support checkpoint resuming (including optimizer, options etc) by default.
- support config file *.yaml and command line config at the same time. (see train_net.sh for example)

The purpose of this code base is you don't need to write those same things again and again for different projects. So I make this code base as general as possible. All you need to change is the network defination and dataset defination in `train.py`.

You can also add your task-specific default configs to `lib/common/configs.py`, and use either *.yaml or command lines to overwrite those configs. (see train_net.sh for example)



