import os
import shutil
import sys

data_dir = sys.argv[1]

new_dir = "./local_dataset"

if os.path.isdir(new_dir):
    shutil.rmtree(new_dir)


def make_dirs(data_dir, new_dir, data_subdir, new_subdir):

    new_path = os.path.join(new_dir, new_subdir)
    old_path = os.path.join(data_dir, data_subdir)

    shutil.copytree(old_path, new_path)


make_dirs(data_dir, new_dir, "train/hazy", "trainA")
make_dirs(data_dir, new_dir, "train/GT", "trainB")
make_dirs(data_dir, new_dir, "val/hazy", "testA")
make_dirs(data_dir, new_dir, "val/GT", "testB")
