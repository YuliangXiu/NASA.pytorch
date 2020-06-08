CUDA_VISIBLE_DEVICES=1 python train_net.py -cfg ../configs/example.yaml -- \
    learning_rate 0.01 \
    batch_size 64 \
    momentum 0.5 \
    optim SGD \
    num_epoch 3 \
    freq_eval 900