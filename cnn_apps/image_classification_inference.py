import torch
from cnn_apps.image_classification import ImageClassificationNet
from cnn_apps.indoor_dataset import IndoorDataSet
from torch.utils.data import DataLoader
import os
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

classes = sorted(os.listdir('/home/charan/Documents/research/dataset/indoor-scenes-cvpr-2019/indoorCVPR_09/Images/'))
root_path = '/home/charan/Documents/research/dataset/indoor-scenes-cvpr-2019'

test_indoor = IndoorDataSet(root_path, classes, train=False, transform_spec=data_transforms['test'])

test_data_loader = DataLoader(test_indoor, batch_size=1, shuffle=False, num_workers=1)

saved_model = ImageClassificationNet()
saved_model.load_state_dict(torch.load('v1_indoor.pth'))
saved_model.eval()

if torch.cuda.is_available():
    saved_model.cuda()

for i, (label, image) in enumerate(test_data_loader):
    if torch.cuda.is_available():
        label, image = label.cuda(), image.cuda()
    predicted = saved_model(image)
    _, index = torch.max(predicted, 1)
    print(f'{label} - {index}')
    print(f'actual class - {classes[label]} , predicted class {classes[index]}')
    print('predicted.')
