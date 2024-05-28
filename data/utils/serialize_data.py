# Simple data serialization for fast data loading during model training
# Takes in images and projects and converts into a tflite file for use with a DataLoader

import tensorflow as tf
from PIL import Image
import os

def _toBytes(image):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(image).numpy()]))

def _toFloat64(*values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[*values]))

# 
def serialize(imagePath, labelPath):
    image = Image.open(imagePath)
    feature = {
        'image': _toBytes(image)
    }

    with open(labelPath, 'r') as labels:
        for line in labels.readlines():
            label = line.strip().split()
            feature[label[0]] = _toFloat64(float(label[1]), float(label[2]))

    proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return proto.SerializeToString()

def main():
    dataset_path = os.path.dirname(os.getcwd())
    projections_data = os.path.join(dataset_path, "annotated_frames")
    projections_labels = os.path.join(dataset_path, "projections_2d")
    writer = tf.io.TFRecordWriter('dataset.tfrecords')

    for i in os.listdir(projections_labels): # data_0, data_1, etc
        print("Reading:", i)
        data_jpg = [f for f in os.listdir(os.path.join(projections_data, i)) if f.lower().endswith(".jpg")]
        for img in data_jpg:
            tok1, _, tok2 = img.split('_')
            writer.write(serialize(os.path.join(projections_data, i, img), os.path.join(projections_labels, i, tok1+"_jointsCam_"+tok2.split('.')[0]+".txt")))
        print("serialized:", i)
    writer.close()
            

if __name__ == '__main__':
    main()
