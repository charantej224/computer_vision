from datasets.coco_dataset import CocoDataset

if __name__ == "__main__":
    coco_dataset = CocoDataset(root_directory="/home/charan/Documents/research/dataset/ms_coco")
    img, target = coco_dataset[20]
    print(target)
