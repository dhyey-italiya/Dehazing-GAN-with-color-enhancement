set -ex
python3 train.py --dataroot ./local_dataset --gpu_ids -1 --name p2p --model pix2pix --save_epoch_freq 1 --save_latest_freq 3500 --display_id -1 --direction AtoB  --lr_decay_iters 500
