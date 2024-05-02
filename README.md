**Important instruction in this document is written in bold, please read them carefully.**

# Image Dehazing and Color Reconstruction Using Pix2Pix GAN

## Dhyey Italiya (2021A7PS1463P)

## Abir Abhyankar (2021A7PS0523P)

This deep learning project aims to dehaze the hazy images employing Leaning based approach,i.e. Pix2Pix GAN.

## Getting Started

To get started with this project, follow the instructions below.

### Prerequisites

Please ensure you have Python installed on your system. We are using Python 3.9.7 for the project. Also ensure your system has:

- torch>=1.4.0
- torchvision>=0.5.0
- dominate>=2.4.0
- visdom>=0.1.8.8
- numpy >=1.20.3
- scikit-image>=0.18.3

Run the script import_libs.py by running the following command in the terminal:
`python3 import_libs.py`

Once you have installed the necessary libraries, you need to generate the dataset in the form that the model requires.
**-Ensure you are inside the directory having the model files. Now run the script prepare_data.sh by the command.**
**-Also ensure that the original dataset is not in the same folder as model files.This might create conflicting directories**
-The original dataset must be in the form of structure(including names of the directories):

--final_dataset
+--train
| +--hazy
| +--GT
+--val
+--hazy
+--GT

`sh prepare_data.sh "<DATAPATH>"`
**Ensure that the datapath to the folder `final_dataset` is written in quotation marks while passing it as command line arguments**

You could see that a folder named `local_dataset` with the file structure:

--local_dataset
+--trainA
+--trainB
+--testA
+--testB

**Here trainA contains Hazy training images, trainB contains GT training images, testA contains Hazy testing images and testB contains GT testing images.**

## Training

- If you have GPUs available and want to use them, then inside train.sh change the --gpu_ids -1 to --gpu_ids <ids> where ids is the GPU ids(separated by comma if multiple available) that you want to use.

- Ensuring a directory named `local_dataset` is created in the directory containing model files. Run the following command:

`sh train.sh`

The model should start training itself.

## Evaluation

Open the file `testing_code.py` and change the paths as required and run the file by typing the below command in the terminal:

`python3 testing_code.py`

- TEST_DATASET_GT_PATH is not necessary but if you want to calculate the SSIN and PSNR after generating the images then add the path to Ground truth Testing images.
