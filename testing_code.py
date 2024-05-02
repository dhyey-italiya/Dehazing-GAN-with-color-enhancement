TEST_DATASET_HAZY_PATH = "/Users/abirabh/Documents/hazy_folder"
TEST_DATASET_OUTPUT_PATH = "/Users/abirabh/Desktop/dehazed_output"
TEST_DATASET_GT_PATH = "/Users/abirabh/Downloads/gt"

import os
from models.base_model import *
from models.networks import *
from models.pix2pix_model import *
from util.util import *
import torch
from PIL import Image
import random
import torchvision.transforms as transforms
from SSIMPSNR import compute_psnr, compute_mssim
from decimal import Decimal

device = "cuda" if torch.cuda.is_available() else "cpu"


def params(size):
    w, h = size
    new_h = h
    new_w = w
    new_h = new_w = 256
    x = random.randint(0, np.maximum(0, new_w - 256))
    y = random.randint(0, np.maximum(0, new_h - 256))
    flip = False
    return {"crop_pos": (x, y), "flip": flip}


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if ow > tw or oh > th:
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def transform(params):
    transform_list = []
    osize = [256, 256]
    transform_list.append(
        transforms.Resize(osize, transforms.InterpolationMode.BICUBIC)
    )
    transform_list.append(
        transforms.Lambda(lambda img: __crop(img, params["crop_pos"], 256))
    )
    if params["flip"]:
        transform_list.append(
            transforms.Lambda(lambda img: __flip(img, params["flip"]))
        )
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __patch_instance_norm_state_dict(state_dict, module, keys, i=0):
    """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
    key = keys[i]
    if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
        if module.__class__.__name__.startswith("InstanceNorm") and (
            key == "running_mean" or key == "running_var"
        ):
            if getattr(module, key) is None:
                state_dict.pop(".".join(keys))
        if module.__class__.__name__.startswith("InstanceNorm") and (
            key == "num_batches_tracked"
        ):
            state_dict.pop(".".join(keys))
    else:
        __patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)


def get_model():
    model = define_G(3, 3, 64, "unet_256", "batch", True, "normal", 0.02, []).to(device)

    state_dict = torch.load(
        "checkpoints/p2p3/latest_net_G.pth", map_location=torch.device(device)
    )

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    if hasattr(state_dict, "_metadata"):
        del state_dict._metadata

    for key in list(state_dict.keys()):
        __patch_instance_norm_state_dict(state_dict, model, key.split("."))

    model.load_state_dict(state_dict)

    return model


def display_stats():
    res_dir = TEST_DATASET_OUTPUT_PATH

    ref_dir = TEST_DATASET_GT_PATH

    print("Computing PSNR and SSIM...")

    ref_pngs = [p for p in os.listdir(ref_dir)]
    res_pngs = [p for p in os.listdir(res_dir)]

    ref_pngs.sort(key=lambda x: int(x.split(".")[0]))
    res_pngs.sort(key=lambda x: int(x.split(".")[0]))
    # if not (len(ref_pngs)==5 and len(res_pngs)==5):
    # raise Exception('Expected 5 .png images, got %d'%len(res_pngs))

    scores = []
    scores_ssim = []
    data = zip(ref_pngs, res_pngs)
    c = 0
    for ref_im, res_im in np.array(list(data)):
        # print(
        #     ref_im,
        #     res_im,
        #     "psnr:",
        #     compute_psnr(ref_im, res_im),
        #     "ssim:",
        #     compute_mssim(ref_im, res_im),
        # )
        scores.append(compute_psnr(ref_dir, res_dir, ref_im, res_im))
        scores_ssim.append(compute_mssim(ref_dir, res_dir, ref_im, res_im))
        c += 1
        if (c % 200) == 0:
            print(f"{c} images processed")
    # print(ref_im, res_im)

    # print(scores[-1])
    psnr = np.mean(scores)
    psnr = Decimal(psnr).quantize(Decimal("0.0000"))
    mssim = np.mean(scores_ssim)
    mssim = Decimal(mssim).quantize(Decimal("0.0000"))
    print("\nPSNR:", psnr, "\nSSIM:", mssim)


os.makedirs(TEST_DATASET_OUTPUT_PATH, exist_ok=True)

input_images = os.listdir(TEST_DATASET_HAZY_PATH)
input_images.sort(key=lambda x: int(x.split(".")[0]))
num = len(input_images)
output_images = []

for i in range(num):
    output_images.append(os.path.join(TEST_DATASET_OUTPUT_PATH, input_images[i]))
    input_images[i] = os.path.join(TEST_DATASET_HAZY_PATH, input_images[i])

"""
Write the code here to load your model
"""

model = get_model()

im_transform = None

# model.eval()

print("Inferencing...")

for i in range(num):
    # print(input_images[i])
    if (i + 1) % 100 == 0:
        print(f"{i+1} images processed")
    # continue
    im_path = input_images[i]
    im = Image.open(im_path).convert("RGB")
    tr_par = params(im.size)
    im_transform = transform(tr_par)
    im = im_transform(im)
    im = torch.FloatTensor([im.tolist()])
    im = im.to(device)
    # print(im)
    with torch.no_grad():
        dehazed_image = model(im)
        # print(dehazed_image)
        # break
        im_np = tensor2im(dehazed_image)
        # print(im_np)
        save_image(im_np, output_images[i])
    # now save the dehazed image at the path indicated by output_images[i]

print("Results have been stored in your results folder.")

if TEST_DATASET_GT_PATH != "":
    display_stats()
else:
    print(
        "Please enter the path for the Ground Tuth Directory,\nto compute PSNR and SSIM for your test results."
    )
