import os

root_path = '/home/charan/Documents/research/dataset/indoor-scenes-cvpr-2019'
root_path = os.path.join(root_path, 'TrainImages.txt')

with open(root_path, 'r') as f:
    ids = [each_id.strip() for each_id in f.readlines()]
    f.close()

print(len(ids))

from PIL import Image

img = Image.open(
    '/home/charan/Documents/research/dataset/indoor-scenes-cvpr-2019/indoorCVPR_09/Images/winecellar/bodega_12_11_flickr.jpg')

import torchvision.transforms as transforms

img = transforms.ToTensor()(img).unsqueeze(0)

print(img)
