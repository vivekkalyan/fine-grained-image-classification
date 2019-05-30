import torchvision
import torch.nn as nn

class Resnet34(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        model = torchvision.models.resnet34(pretrained=True)
        in_features = model.fc.in_features
        self.resnet = nn.Sequential(*(list(model.children())[:-1]))
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, inputs):
        x = inputs['image']
        x = self.resnet(x)
        x = x.squeeze(dim=3).squeeze(dim=2)
        x = self.fc(x)
        return x

    def freeze(self):
        for param in self.resnet.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.resnet.parameters():
            param.requires_grad = True
