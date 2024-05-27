from torchvision.io import read_image
from torch.utils.data import Dataset

class HandPose(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        