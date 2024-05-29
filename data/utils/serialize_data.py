# Simple data serialization for fast data loading during model training
# Takes in images and projects and converts into a tflite file for use with a DataLoader

from torchvision.transforms import v2
import tensorflow as tf
from PIL import Image
import os
import torch

transforms = v2.Compose([
    v2.PILToTensor(),
    v2.Resize(size=(224, 224), antialias=True),
    v2.ToDtype(torch.float32),
    v2.Normalize(mean=[0.43625, 0.41656, 0.43584], std=[0.24982, 0.24742, 0.26399])
])

def _toBytes(image):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(image).numpy()]))

def _toFloat(*values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[*values]))


def serialize(imagePath, labelPath):
    image = Image.open(imagePath)

    base_width = 224
    wpercent = (base_width / float(image.size[0]))
    hsize = int((float(image.size[1]) * float(wpercent)))
    image = image.resize((base_width, hsize), Image.Resampling.LANCZOS)

    feature = {
        'image': _toBytes(image)
    }

    with open(labelPath, 'r') as labels:
        for line in labels.readlines():
            label = line.strip().split()
            feature[label[0]] = _toFloat(float(label[1]), float(label[2]))

    proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return proto.SerializeToString()

def main():
    dataset_path = os.path.dirname(os.getcwd())
    projections_data = os.path.join(dataset_path, "annotated_frames")
    projections_labels = os.path.join(dataset_path, "projections_2d")
    writer = tf.io.TFRecordWriter('dataset.tfrecords')
    # mean = 0.0
    # std = 0.0
    # nb_samples = 0


    for i in os.listdir(projections_labels): # data_0, data_1, etc
        print("Reading:", i)
        data_jpg = [f for f in os.listdir(os.path.join(projections_data, i)) if f.lower().endswith(".jpg")]
        for img in data_jpg:
            tok1, _, tok2 = img.split('_')
            # img = Image.open(os.path.join(projections_data,i, img))
            # img = transforms(img)
            # img = img.view(1, 3, -1)
            # mean += img.mean(2).sum(0)
            # std += img.std(2).sum(0)
            # nb_samples += 1
            writer.write(serialize(os.path.join(projections_data, i, img), os.path.join(projections_labels, i, tok1+"_jointsCam_"+tok2.split('.')[0]+".txt")))
        print("serialized:", i)

    # mean /= nb_samples
    # std /= nb_samples
    # print("Mean:", mean)
    # print("Standard Deviation:", std)
    writer.close()
            

if __name__ == '__main__':
    main()
