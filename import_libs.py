import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

packages = [
    "torch",
    "torch.optim",
    "torchvision",
    "matplotlib",
    "scikit-image",
    "pillow",
    "dominate",
    "visdom",
    "numpy",
    "wandb"
]

for package in packages:
    install(package)
