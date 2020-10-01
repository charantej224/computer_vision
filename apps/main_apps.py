from datasets.coco_dataset import CocoDataset
from torch.utils.data import DataLoader

coco_dataset_train = CocoDataset(root_directory="/home/charan/Documents/research/dataset/ms_coco")
coco_dataset_test = CocoDataset(root_directory="/home/charan/Documents/research/dataset/ms_coco",
                                annotation_dir="annotations/test_2017.json")

data_loader_train = DataLoader(
    coco_dataset_train, batch_size=2, shuffle=True, num_workers=4)

data_loader_test = DataLoader(
    coco_dataset_test, batch_size=1, shuffle=False, num_workers=4)


# Testing Dataset.
# if __name__ == "__main__":
#     coco_dataset = CocoDataset(root_directory="/home/charan/Documents/research/dataset/ms_coco")
#     img, target = coco_dataset[20]
#     print(target)
