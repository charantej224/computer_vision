import torch.nn as nn


class ImageClassificationNet(nn.Module):
    def __init__(self):
        super(ImageClassificationNet, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.cnn2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(64 * 56 * 56, 1000)
        self.fc2 = nn.Linear(1000, 67)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.cnn2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
