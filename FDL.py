"""
# The following code was written to be run on google colab, using Google Drive as data storage.
# Requirements, libraries
!pip install ternausnet > /dev/null
!pip install pillow
!pip install torchsummary
!pip install segmentation_models_pytorch
"""

from collections import defaultdict
import copy
import random
import os
import pandas as pd
import numpy as np
import shutil
from urllib.request import urlretrieve
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
import ternausnet.models
from tqdm import tqdm
import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import tensorflow as tf
from tensorflow.python.client import device_lib
from torch.utils.data import Dataset, DataLoader
from google.colab import drive
from PIL import Image
import torchsummary
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import PSPNet
import time
import torch.nn.functional as F
from PIL import Image
import os
from torchvision.transforms import functional as TF
import tarfile

cudnn.benchmark = True

"""
# Check what version of Python we are running
!python3 -V
!uname -m
"""

# check whether GPU is active
print(tf.test.gpu_device_name())
device_lib.list_local_devices()

# Obtaining the data

drive = drive.mount('/content/drive')

# Define the path for raw folder and masks folder 
# set the owner of google drive repository (in this case baso or leo)
who = "baso"

if (who == "leo"):
  masks_folder_path = '/content/drive/MyDrive/deep_learning/kaggl/vectors/random-split-_2022_11_17-22_35_45/Masks'
  raw_folder_path = '/content/drive/MyDrive/deep_learning/kaggl/rasters/raw'
  path_pred = '/content/drive/MyDrive/deep_learning/kaggl/rasters/predictions'
elif(who == "baso"):
  masks_folder_path = '/content/drive/MyDrive/Hurricane_Harvey/vectors/random-split-_2022_11_17-22_35_45/Masks'
  raw_folder_path = '/content/drive/MyDrive/Hurricane_Harvey/rasters/raw'
  prediction_folder = '/content/drive/MyDrive/deep_learning/kaggl/rasters/raw_prediction'
  path_pred = '/content/drive/MyDrive/Hurricane_Harvey/predictions2'


# Create an empty list to store the results
test_pics=[]
train_pics=[]

# Get a list of all files in folder B
files_b = os.listdir(raw_folder_path)

# Iterate through the files in folder B
for file_b in files_b:
    # Get the file name without the file format
    file_name, file_extension = os.path.splitext(file_b)
    # Check if a file with the same name and .png extension exists in folder A
    if os.path.isfile(os.path.join(masks_folder_path, file_name + '.png')):
        train_pics.append(file_name)
    else:
        test_pics.append(file_name)
print(len(files_b))
print(len(test_pics))
print(len(train_pics))

"""
Once the raw images are stored in one place, we create a variable to iterate through them. 
To do so, we generate a list, sort the list, and then shuffle the list. This helps us ensure that there is 
no entry bias.  We also check the amount of pictures we have, to make sure that no mistakes were made.
"""

# Create a list of raw images in tif format
images_filenames = list(sorted(os.listdir(raw_folder_path)))
print("In total, there are 374 images available to work with "+str(len(images_filenames)))

random.seed(42)
random.shuffle(images_filenames)

# We keep 20% of our training set for validation purposes.

# splitting training data in train_set (80%) and val_set (20%)
train_set = train_pics[0:240]
print("Our train set contains "+str(len(train_set))+" images.")

val_set = train_pics[241:300]
print("Our validation set contains "+str(len(val_set))+" images.")

"""
We then create a function called "display_image_grid" that takes in several arguments: "images_filenames", 
"images_directory", "masks_directory", "predicted_masks", "batch_size", and "size". 
The function creates a grid of images using the matplotlib library and displays them using the OpenCV library.

The function starts by defining the number of columns in the grid, which is 2 if "predicted_masks" is not provided, 
and 3 if it is. It then sets up a loop to iterate through the images in "images_filenames" in batches of size "batch_size". 
For each batch, it creates a new figure with a specific number of rows and columns using the "plt.subplots()" function.

It then iterates through the images in the current batch and reads the image and mask files using the OpenCV library. It then resizes the image and mask to the given "size" using the "cv2.resize()" function. It then uses the "imshow()" function from OpenCV to display the images in the grid. The first column is the original image, the second column is the ground-truth mask, and the third column is the predicted mask (if provided). The function also includes axis labels and titles for each column. Finally, it shows the grid using "plt.show()" and pauses for 0.5 seconds before closing the plot and moving on to the next batch of images.
"""

def display_image_grid(images_filenames, images_directory, masks_directory, predicted_masks=None, batch_size=10, size=(256,256)):
    cols = 3 if predicted_masks else 2
    rows = len(images_filenames)
    k = 0
    
    # Create a figure with a specific number of rows and columns

    for i in range(0,rows,batch_size):
        figure, ax = plt.subplots(nrows=batch_size, ncols=cols, figsize=(10, 24))
        for j, image_filename in enumerate(images_filenames[i:i+batch_size]):
            k = k + 1
            image_path = (os.path.join(images_directory, image_filename) + ".tif")
            mask_path = (os.path.join(masks_directory, image_filename) + ".png")

            # Read the image and reduce its size
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(cv2.resize(image,size), cv2.COLOR_BGR2RGB)

            # Read the mask and reduce its size
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            mask = cv2.resize(mask,size)

            # Use the imshow() function from OpenCV to display the images
            ax[j, 0].imshow(image)
            ax[j, 1].imshow(mask, interpolation="nearest")

            ax[j, 0].set_title("Image")
            ax[j, 1].set_title("Ground truth mask")

            ax[j, 0].set_axis_off()
            ax[j, 1].set_axis_off()

            if predicted_masks:
                predicted_mask = predicted_masks[i+j]
                ax[j, 2].imshow(predicted_mask, interpolation="nearest")
                ax[j, 2].set_title("Predicted mask")
                ax[j, 2].set_axis_off()
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()
    print("number of plots:" + str(k))
    
"""
WARNING: The function below shows the train pics and their associated masks. 
It outputs close to 300 images and their masks in parallel.
As so, please consider that running the function below will take a lot of space in memory.
"""

print(len(train_pics))

display_image_grid(train_pics, raw_folder_path, masks_folder_path)

"""
Here we create a custom dataset class that inherits from the PyTorch "Dataset" class. 
This class is here to load and preprocess images and masks that are specific to the Hurricane Harvey dataset.

The class has a constructor "init" which takes in three arguments: "images_filenames", "images_directory", "masks_directory" 
and "transform". 
"images_filenames" is a list of image file names, "images_directory" is the directory where the images are located, 
"masks_directory" is the directory where the masks are located, and "transform" is an optional argument that can be used 
to apply data augmentation to the images and masks.

"len" and "getitem" are required by PyTorch's dataset framework. "len" returns the number of images in the dataset and "getitem" is used to access a specific image and its corresponding mask.

The "getitem" method takes the index of the image and reads the image using OpenCV library. It then converts the image from BGR to RGB color space. It also reads the corresponding mask file using OpenCV and converts it from 8-bit integer to float32. Finally, it applies transformations to the image and mask, if there is a mask provided.
"""

class Hurricane_Harvey(Dataset):
    def __init__(self, images_filenames, images_directory, masks_directory, transform=None):
        self.images_filenames = images_filenames
        self.images_directory = images_directory
        self.masks_directory = masks_directory
        self.transform = transform

    # Returns the length of some set
    def __len__(self):
        return len(self.images_filenames)

    # This accesses a specific images/mask
    def __getitem__(self, idx):
        image_filename = self.images_filenames[idx]
        image_path = (os.path.join(self.images_directory, image_filename) + ".tif")
        #print(image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_path = (os.path.join(self.masks_directory, image_filename) + ".png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE,)
        #mask = mask.astype(np.uint8)
        mask = mask.astype(np.float32)
        #print(mask)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask

"""
This code is defining two sets of image preprocessing steps, one for training data (train_transform) and one for validation data (val_transform). 
The training data preprocessing includes several image augmentation techniques such as resizing, shifting, rotating, color shifting, and adjusting brightness and contrast. 

These techniques are applied randomly with a probability of 0.5 for each step. The final step in the training preprocessing is normalizing 
the image by subtracting mean values and dividing by standard deviations and converting the image to a PyTorch tensor using ToTensorV2.

The validation data preprocessing is less extensive, it only includes resizing and normalizing the image by subtracting 
mean values and dividing by standard deviations and converting the image to a PyTorch tensor using ToTensorV2.

The train_dataset and val_dataset are created by using the Hurricane_Harvey class which is custom dataset class and passing 
the respective set of images(train_set, val_set), the raw image folder path, mask folder path and the respective transform functions.
"""

train_transform = A.Compose(
    [
        A.Resize(480, 640), #resize images to 480*640 pixels
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5), #randomly shift, scale and rotate images
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5), #randomly change colors
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5), #adjust brightness and contrast
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), #Normalize pixel values
        ToTensorV2(), #convert to pytorch tensor
    ]
)

train_dataset = Hurricane_Harvey(train_set, raw_folder_path, masks_folder_path, transform=train_transform,)

val_transform = A.Compose(
    [A.Resize(480, 640), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
)
val_dataset = Hurricane_Harvey(val_set, raw_folder_path, masks_folder_path, transform=val_transform,)

"""
This code defines a function called "visualize_augmentations" which takes in three inputs: a dataset, an index of an image 
in the dataset, and the number of samples. The function creates a subplot with the specified number of rows and columns, 
and plots the original and augmented images using the Matplotlib library.

The first line of the function makes a copy of the input dataset and removes the normalization and conversion to tensor steps 
from the dataset's transform attribute. This is done so that the visualized images will be in their original format and not normalized.

The for loop iterates for the specified number of samples. In each iteration, it selects the image and mask from the dataset based 
on the specified index and plots the image and mask using the Matplotlib's imshow method. The titles of the images and masks are also set, and the axis is turned off.

The function then shows the plot using the show method of Matplotlib and tightens the layout of the plot.
It allows to visualize the image and mask with the augmentations applied to them and is useful for debugging, 
quality control and understanding the effect of the applied augmentations.
"""

def visualize_augmentations(dataset, idx=0, samples=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    figure, ax = plt.subplots(nrows=samples, ncols=2, figsize=(10, 24))
    for i in range(samples):
        image, mask = dataset[idx]
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(mask, interpolation="nearest")
        ax[i, 0].set_title("Augmented image")
        ax[i, 1].set_title("Augmented mask")
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
    plt.tight_layout()
    plt.show()

random.seed(42) # the images are in random order
visualize_augmentations(train_dataset, idx=55)

"""
In the train function, the images and target variables are used as inputs to the model and the criterion function.

images: The images variable is used to store a batch of images from the train_loader, which is passed to the model. These images are used as input to the model to make predictions.

target: The target variable is used to store the corresponding ground-truth labels or segmentation masks for the images. 
These are used as the correct output for the model, and the criterion function uses these to calculate the loss between the model's predictions and the target values.

In this code, the images and targets are first moved to the appropriate device using the to method, based on the device specified in the params dictionary.
The model is then applied to the images, and the output is obtained. The criterion function is applied to the output and the target, 
which calculates the loss between the model's predictions and the correct outputs. This loss is then used to update the model's parameters using the optimizer.

In summary, the images and target variables are used to train the model, the model takes images as input, 
and the criterion function uses the target as the correct output to calculate the loss and update the model's parameters.
"""

# params for the model
params = {
    "lr": 0.001,
    "batch_size": 4,
    "num_workers": 1,
    "epochs": 10,
}

# Running the model
# reccomendation max_lr = 1e-3

#first PSPNet resnet101 submission  Score: 70.32
#params = {
    #"lr": 0.001,
    #"batch_size": 4,
    #"num_workers": 1,
    #"epochs": 10,
#}

#second PSPNet resnet101 submission Score: 14.39
#
#params = {
#    "lr": 0.0001,
#    "batch_size": 4,
#    "num_workers": 1,
#    "epochs": 20,
#}

#third PSPNet resnet34 submission Score: 70.25

#params = {
#    "lr": 0.001,
#    "batch_size": 4,
#    "num_workers": 1,
#    "epochs": 5,
#}

#fourth PSPNet resnet34 submission Score: 66.06

#params = {
    #"lr":  0.000001,
    #"batch_size": 4,
    #"num_workers": 1,
    #"epochs": 25,
#}

#fifth PSPNet resnet34 submission Score: 70.63

#params = {
#    "lr": 0.001,
#    "batch_size": 4,
#    "num_workers": 1,
#    "epochs": 10,
#}

model = smp.PSPNet(
    encoder_name = 'resnet34',
    encoder_weights = 'imagenet',
    classes = 27,
    activation = 'softmax2d', # could be None for logits or 'softmax2d' for multiclass segmentation
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_decay = 1e-4
patience = 5
criterion = nn.CrossEntropyLoss().to(device)
dice_loss = smp.losses.DiceLoss(mode='multiclass', classes=27)
optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

def dataloaders(model, train_dataset, val_dataset, params):
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader

train_loader, val_loader = dataloaders(model, train_dataset, val_dataset, params)

dice_loss = smp.losses.DiceLoss(mode='multiclass', classes=27)

def get_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output,dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, patience):
    torch.cuda.empty_cache()
    total_train_loss = []
    total_val_loss = []
    train_acc_log = []
    train_dice_log = []
    train_loss_log = []
    val_acc_log = []
    val_loss_log = []
    lrs = []
    best_val_dice = 0
    train_dice_tot = 0
    val_dice_tot = 0
    early_stopping_counter = 0

    model.to(device)
    for e in range(epochs):
        print(f'epoch: {e}')
        train_loss = []
        running_loss = 0
        accuracy = 0
        train_dice = []
        model.train()
        for i, batch in enumerate(tqdm(train_loader)):
            #training phase
            optimizer.zero_grad()
            images, masks = batch
            images, masks = images.to(device), masks.to(device)
            masks = masks.type(torch.LongTensor)
            output = model(images.to(device))
            loss = criterion(output.float().to(device), masks.to(device))
            dice = dice_loss(output.float().to(device), masks.to(device))
            dice_s = 1 - dice.item()
            accuracy += get_accuracy(output.float().to(device), masks.to(device))
            loss.backward()
            optimizer.step()
            train_dice.append(dice_s)
            running_loss += loss.item()
            train_loss.append(loss.item())

        train_acc = (accuracy / len(train_loader))
        train_dice_tot = np.mean(train_dice)
        train_acc_log.append(train_acc)
        train_loss_mean = np.mean(train_loss)
        train_loss_log.append(train_loss_mean)


        model.eval()
        val_losses = []
        running_val_loss = 0
        val_accuracy = 0
        val_dice = []
        #validation loop
        with torch.no_grad():
            for i, val_batch in enumerate(tqdm(val_loader)):
                images, masks = val_batch
                images, masks = images.to(device), masks.to(device)
                masks = masks.type(torch.LongTensor)
                output = model(images.to(device))
                loss = criterion(output.float().to(device), masks.to(device))
                dice = dice_loss(output.float().to(device), masks.to(device))
                dice_s = 1 - dice.item()
                val_dice.append(dice_s)
                val_accuracy += get_accuracy(output.float().to(device), masks.to(device))
                running_val_loss += loss.item()
                val_losses.append(loss.item())

        val_acc = (val_accuracy / len(val_loader))
        val_acc_log.append(val_acc)
        val_dice_tot = np.mean(val_dice)
        val_loss_mean = np.mean(val_losses)
        val_loss_log.append(val_loss_mean)

        print(f'Train Accuracy: {train_acc}\nTrain Dice: {train_dice_tot}\nTrain Loss: {train_loss_mean}\nValidation Accuracy: {val_acc}\nValidation Dice: {val_dice_tot}\nValidation Loss: {val_loss_mean}')
        log = {"train_acc": train_acc_log, "val_acc": val_acc_log, "train_loss": train_loss_log, "val_loss": val_loss_log}

        # Save the model if it has improved
        if val_dice_tot > best_val_dice:
            best_val_dice = val_dice_tot
            torch.save(model.state_dict(), 'model_APT_ss.pt')
            torch.save(model, 'model_APT.pt') # add name of model # add name of model
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        # If the counter reaches the patience, stop training
        if early_stopping_counter >= patience:
            print("Early stopping")
            break

        torch.cuda.empty_cache()

    log = {"train_acc": train_acc_log, "val_acc": val_acc_log, "train_loss": train_loss_log, "val_loss": val_loss_log}

    return log

fit(params['epochs'], model, train_loader, val_loader, criterion, optimizer, scheduler, patience)

torchsummary.summary(model, (3,480,640))

class Hurricane_Harvey_inference(Dataset):
    def __init__(self, images_filenames, images_directory, transform=None):
        self.images_filenames = images_filenames
        self.images_directory = images_directory
        self.transform = transform

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        image_filename = self.images_filenames[idx]
        image_path = (os.path.join(self.images_directory, image_filename) + ".tif")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = tuple(image.shape[:2])
        #image = cv2.resize(image, (256, 256))
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        #print(type(image))
        return image, original_size, image_filename

# Testing the model

# Here we run our transformations on the test pics instead

test_transform = A.Compose(
    [
        A.Resize(480, 640),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)
test_dataset = Hurricane_Harvey_inference(test_pics, raw_folder_path, transform=test_transform,)

def predict(model, params, test_dataset, batch_size):
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=params["num_workers"], pin_memory=True,
    )
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, (original_heights, original_widths), image_filename in test_loader:
            images = images.to(device, non_blocking=True)
            output = model(images)
            #print(len(torch.unique(output)))
            probabilities = torch.argmax(output, dim=1)
            #print(torch.unique(probabilities)) # to see if the output of probabilites is composed by different numbers
            predicted_masks = probabilities.cpu().squeeze(0)
            predicted_masks = predicted_masks.cpu().numpy().astype(np.uint8)
            i = 0
            for predicted_mask, original_height, original_width in zip(
                predicted_masks, original_heights.numpy(), original_widths.numpy()
            ):
                predictions.append((predicted_mask, original_height, original_width, image_filename[i]))
                i = i + 1
    return predictions

predictions = predict(model, params, test_dataset, batch_size=4)
# the printed output you see here is to see if the probabilietis variable in
# predict funciton is producing tensors with different values
# in this case is producing tensors with only 0s in them



# Create folder to save masks
if not os.path.exists(path_pred):
    os.mkdir(path_pred)

# Iterate through predictions
for (predicted_mask, original_height, original_width, image_filename) in predictions:
    # Resize the mask to the original dimensions
    predicted_mask = cv2.resize(predicted_mask, (original_width, original_height), interpolation = cv2.INTER_NEAREST)
    # Convert mask to PIL image
    predicted_mask = Image.fromarray(predicted_mask)
    predicted_mask = predicted_mask.convert("L")
    # Save mask to folder
    save_path = os.path.join(path_pred, f"{image_filename}.png")
    predicted_mask.save(save_path, "PNG")
 

# tar submission

tar = tarfile.open("submissionPSPNet5.tar", "w")

for root, dir, files in os.walk(path_pred):
    for  file in files:
        fullpath = os.path.join(root, file)
        tar.add(fullpath, arcname=file)

tar.close()
