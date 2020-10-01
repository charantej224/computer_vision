import os
from utils.read_files import read_json_value
from PIL import Image
from torchvision.datasets import VisionDataset
import torch


class CocoDataset(VisionDataset):
    def __init__(self, root_directory, annotation_dir="annotations/train_2017.json", image_dict="val2017",
                 transforms=None):
        self.root_directory = root_directory
        self.annotations = read_json_value(os.path.join(root_directory, annotation_dir))
        self.image_path = os.path.join(root_directory, image_dict)
        self.category_dict = self.generate_category_dict()

    def __getitem__(self, index):
        file_name, image_id = self.annotations['images'][index]['file_name'], self.annotations['images'][index]['id']
        image_annotations = filter(lambda x: x['image_id'] == image_id, self.annotations['annotations'])
        file_name = os.path.join(self.image_path, file_name)
        img = Image.open(file_name).convert('RGB')
        boxes = []
        labels = []
        for each_annotation in image_annotations:
            boxes.append([each_annotation["bbox"][0], each_annotation["bbox"][1],
                          each_annotation["bbox"][0] + each_annotation["bbox"][2],
                          each_annotation["bbox"][1] + each_annotation["bbox"][3]])
            labels.append(each_annotation["category_id"])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": image_id}
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.annotations['images'])

    def generate_category_dict(self):
        category_dict = {}
        for each in self.annotations['categories']:
            category_dict[each['id']] = each['name']
        return category_dict

    def get_category_dict(self):
        return self.category_dictf
