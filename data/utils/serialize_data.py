# Simple data serialization for fast data loading during model training
# Takes in images and projects and converts into a tflite file for use with a DataLoader

from torchvision.transforms import v2
from PIL import Image
import cv2
import h5py
import numpy as np
import os
import torch

transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToTensor(),
])



def serialize(imagePath, labelPath):
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_array = transforms(image).permute(1, 2, 0)

    # Create or open an HDF5 file
    with h5py.File('dataset.h5', 'a') as h5file:
        # Create or open datasets to store images and labels
        if 'images' not in h5file:
            images_dataset = h5file.create_dataset('images', (0, *image_array.shape), maxshape=(None, *image_array.shape), dtype='float16')
        else:
            images_dataset = h5file['images']
        if 'labels' not in h5file:
            labels_dataset = h5file.create_dataset('labels', (0,21,2), maxshape=(None,None,None), dtype='float16')
        else:
            labels_dataset = h5file['labels']

        # Append the image and label to the datasets
        images_dataset.resize((images_dataset.shape[0] + 1, *image_array.shape))
        images_dataset[-1] = image_array

        out = []
        with open(labelPath, 'r') as labels:
            for line in labels.readlines():
                label = line.strip().split()
                out.append(np.array([float(label[1]), float(label[2])]))

        labels_dataset.resize((labels_dataset.shape[0] + 1,21, 2))
        labels_dataset[-1] = np.array(out)


def main():
    dataset_path = os.path.dirname(os.getcwd())
    projections_data = os.path.join(dataset_path, "annotated_frames")
    projections_labels = os.path.join(dataset_path, "projections_2d")
    # mean = 0.0
    # std = 0.0
    # nb_samples = 0


    for i in os.listdir(projections_labels): # data_0, data_1, etc
        print("Reading:", i)
        data_jpg = [f for f in os.listdir(os.path.join(projections_data, i)) if f.lower().endswith(".jpg")]
        for img in data_jpg:
            tok1, _, tok2 = img.split('_')
            print("Reading:",os.path.join(projections_data, i, img))
            # img = Image.open(os.path.join(projections_data,i, img))
            # img = transforms(img)
            # img = img.view(1, 3, -1)
            # mean += img.mean(2).sum(0)
            # std += img.std(2).sum(0)
            # nb_samples += 1
            serialize(os.path.join(projections_data, i, img), os.path.join(projections_labels, i, tok1+"_jointsCam_"+tok2.split('.')[0]+".txt"))
        print("serialized:", i)

    # mean /= nb_samples
    # std /= nb_samples
    # print("Mean:", mean)
    # print("Standard Deviation:", std)
            

if __name__ == '__main__':
    main()
