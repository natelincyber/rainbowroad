from torchvision.transforms import v2
from torch.utils.data import Dataset
import torch
import h5py



class HandPose(Dataset):
    def __init__(self, filename="dataset.h5"):
        raw_dataset = h5py.File(filename, 'r')
        self.images = raw_dataset['images']
        self.labels = raw_dataset['labels']

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return torch.from_numpy(self.images[index]), torch.from_numpy(self.labels[index])



if __name__ == '__main__':
    ds = HandPose(filename="C:\\Users\\natel\\Dev\\Projects\\RainbowRoad\\data\\dataset.h5")
    import matplotlib.pyplot as plt

    print(ds[0][0].shape)
    image_np = ds[0][0].to(torch.float32)
    image_np = image_np.permute(1, 2, 0).numpy()
    plt.imshow(image_np)
    plt.show()