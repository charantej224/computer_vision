from torch.utils.data import DataLoader
from torchvision import transforms
from cnn_apps.image_classification import ImageClassificationNet
from cnn_apps.indoor_dataset import IndoorDataSet
import os
import torch.nn as nn
import torch

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

train_indoor = IndoorDataSet(root_path, classes, train=True, transform_spec=data_transforms['train'])
test_indoor = IndoorDataSet(root_path, classes, train=False, transform_spec=data_transforms['test'])

train_data_loader = DataLoader(train_indoor, batch_size=1, shuffle=False, num_workers=1)
test_data_loader = DataLoader(test_indoor, batch_size=1, shuffle=False, num_workers=1)
epochs = 5

model = ImageClassificationNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_list = []
acc_list = []
total_step = len(train_data_loader)

if torch.cuda.is_available():
    model.cuda()

model.train()
for epoch in range(epochs):
    print(f'epoch {epoch} / {epochs}')
    for i, (label, image) in enumerate(train_data_loader):
        # Run the forward pass
        # print(image.shape)
        if torch.cuda.is_available():
            image, label = image.cuda(), label.cuda()

        outputs = model(image)
        loss = criterion(outputs, label)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == label).sum().item()

        if (i + 1) % 100 == 0:
            print(f'Epoch -  {epoch}/{epochs} , loss - {loss.item()}')

torch.save(model.state_dict(), 'v1_indoor.pth')
state_dict = model.state_dict()

for key, value in state_dict.iteritems():
    print(f'{key} - {value}')
