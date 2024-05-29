from torch.utils.data import Dataset
import tensorflow as tf

class HandPose(Dataset):
    def __init__(self, filename="dataset.tfrecords", transform=None):
        self.transform = transform
        self.feature_description = {
            'image': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
            'F4_KNU1_A': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'F4_KNU1_B': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'F4_KNU2_A': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'F4_KNU3_A': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'F4_KNU3_A': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'F3_KNU1_A': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'F3_KNU1_B': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'F3_KNU2_A': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'F3_KNU3_A': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'F1_KNU1_A': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'F1_KNU1_B': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'F1_KNU2_A': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'F1_KNU3_A': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'F2_KNU1_A': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'F2_KNU1_B': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'F2_KNU2_A': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'F2_KNU3_A': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'TH_KNU1_A': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'TH_KNU1_B': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'TH_KNU2_A': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'TH_KNU3_A': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'PALM_POSITION': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
        }

        raw_dataset = tf.data.TFRecordDataset(filename)
        self.dataset = raw_dataset.map(self._parse_dataset)
        self.indexedData = []

        self.prepare_dataset()
        del raw_dataset # send to gc to save memory


    def _parse_dataset(self, proto):
        return tf.io.parse_single_example(proto, self.feature_description)

    def prepare_dataset(self):
        for record in self.dataset:
            self.indexedData.append(record)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.indexedData[index]
        image = data['image'].numpy()
        labels = []
        for item in data:
            if item != 'image':
                labels.append(data[item].numpy())
        return image, labels
        


if __name__ == '__main__':
    ds = HandPose(filename="D:\\Projects\\RainbowRoad\\data\\utils\\dataset.tfrecords")
    print(ds[0])
