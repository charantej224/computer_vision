from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms


class IndoorDataSet(Dataset):
    def __init__(self, root_path, classes, train=True, transform_spec=None):
        self.root_path = root_path
        self.classes = classes
        if transform_spec is None:
            self.transform_spec = transforms.ToTensor()
        else:
            self.transform_spec = transform_spec
        self.ids = []
        if train:
            self.csv_path = os.path.join(self.root_path, 'TrainImages.txt')
        else:
            self.csv_path = os.path.join(self.root_path, 'TestImages.txt')
        print(self.root_path)
        with open(self.csv_path, 'r') as f:
            self.ids = [each_id.strip() for each_id in f.readlines()]
            f.close()

    def __getitem__(self, item):
        image = self.ids[item].split('/')
        print(os.path.join(self.root_path, 'indoorCVPR_09/Images/', self.ids[item]))
        label = self.classes.index(image[0])
        img = Image.open(os.path.join(self.root_path, 'indoorCVPR_09/Images/', self.ids[item]))
        if self.transform_spec is not None:
            img = self.transform_spec(img).unsqueeze(0)
        else:
            img = transforms.ToTensor()(img).unsqueeze(0)
        return label, img

    def __len__(self):
        return len(self.ids)
