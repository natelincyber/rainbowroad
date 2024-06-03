from torchvision.transforms import v2
import cv2
import h5py
import numpy as np
import os

transforms = v2.Compose([
    # v2.Resize(size=(224, 224)),
    v2.ToTensor(),
])

h5file = h5py.File('dataset.h5', 'a')

def serialize(imagePath, labelPath):
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224))
    image_array = transforms(image).permute(2,1,0)

    if 'images' not in h5file:
        images_dataset = h5file.create_dataset('images', (0, *image_array.shape), maxshape=(None, *image_array.shape), dtype='float32', compression="gzip", chunks=True)
    else:
        images_dataset = h5file['images']
    if 'labels' not in h5file:
        labels_dataset = h5file.create_dataset('labels', (0, 21, 2), maxshape=(None, 21, 2), dtype='float32', compression="gzip", chunks=True)
    else:
        labels_dataset = h5file['labels']

    images_dataset.resize((images_dataset.shape[0] + 1, *image_array.shape))
    images_dataset[-1] = image_array

    out = []
    with open(labelPath, 'r') as labels:
        for line in labels.readlines():
            label = line.strip().split()
            out.append(np.array([float(label[1]), float(label[2])], dtype='float32'))

    labels_dataset.resize((labels_dataset.shape[0] + 1, 21, 2))
    labels_dataset[-1] = np.array(out)

def main():
    dataset_path = os.path.dirname(os.getcwd())
    projections_data = os.path.join(dataset_path, "annotated_frames")
    projections_labels = os.path.join(dataset_path, "projections_2d")

    for i in os.listdir(projections_labels):
        print("Reading:", i)
        data_jpg = [f for f in os.listdir(os.path.join(projections_data, i)) if f.lower().endswith(".jpg")]
        for img in data_jpg:
            tok1, _, tok2 = img.split('_')
            print("Reading:", os.path.join(projections_data, i, img))
            serialize(os.path.join(projections_data, i, img), os.path.join(projections_labels, i, tok1+"_jointsCam_"+tok2.split('.')[0]+".txt"))
        print("Serialized:", i)

if __name__ == '__main__':
    main()
