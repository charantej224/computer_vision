from torch.utils.data import DataLoader
from torchvision import transforms

from cnn_apps.indoor_dataset import IndoorDataSet
import os

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

classes = os.listdir('/home/charan/Documents/research/dataset/indoor-scenes-cvpr-2019/indoorCVPR_09/Images/')
root_path = '/home/charan/Documents/research/dataset/indoor-scenes-cvpr-2019'

indoor_dataset = IndoorDataSet(root_path, classes, train=True, transform_spec=None)

data_loader = DataLoader(indoor_dataset, batch_size=1, shuffle=False, num_workers=1)

for label, image in data_loader:
    print(label)
