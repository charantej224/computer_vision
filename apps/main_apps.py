from datasets.coco_dataset import CocoDataset
from torch.utils.data import DataLoader
from models.model_selection import FasterRCNNModel
import torch
from engine import train_one_epoch, evaluate


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

# get the model using our helper function
if __name__ == "__main__":
    model = FasterRCNNModel(num_classes=182)

    # move model to the right device
    model.cuda()

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

