import os

root_path = '/home/charan/Documents/research/dataset/indoor-scenes-cvpr-2019'
root_path = os.path.join(root_path, 'TrainImages.txt')

with open(root_path, 'r') as f:
    ids = [each_id.strip() for each_id in f.readlines()]
    f.close()

print(len(ids))

from PIL import Image

img = Image.open(
    '/home/charan/Documents/research/dataset/indoor-scenes-cvpr-2019/indoorCVPR_09/Images/studiomusic/StudioA1.jpg')

import torchvision.transforms as transforms

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

img = data_transforms['train'](img)

print(img.shape)
