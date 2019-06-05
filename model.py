import torch
import torch.nn as nn
import torchvision

from utils import device

class Resnet34(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        model = torchvision.models.resnet34(pretrained=True)
        in_features = model.fc.in_features
        self.resnet = nn.Sequential(*(list(model.children())[:-2]))
        self.concat_pool = AdaptiveConcatPool2d()
        self.fc1 = LinearLayer(in_features*2, 512, dropout=0.25,
                activation=nn.ReLU())
        self.fc2 = LinearLayer(512, out_features, dropout=0.5)

    def forward(self, inputs):
        x = inputs['image'].to(device())
        x = self.resnet(x)
        x = self.concat_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def freeze(self):
        for param in self.resnet.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.resnet.parameters():
            param.requires_grad = True


class LinearLayer(nn.Module):
    " batch_norm -> dropout -> linear -> activation "
    def __init__(self, in_feat, out_feat, bn=True, dropout=0., activation=None):
        super().__init__()
        layers = []
        if bn: layers.append(BatchNorm1dFlat(in_feat))
        if dropout != 0: layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_feat, out_feat))
        if activation is not None: layers.append(activation)
        self.linear_layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear_layer(x)

class BatchNorm1dFlat(nn.BatchNorm1d):
    "`nn.BatchNorm1d`, but first flattens leading dimensions"
    def forward(self, x):
        if x.dim()==2: return super().forward(x)
        *f, c = x.shape
        x = x.contiguous().view(-1, c)
        return super().forward(x).view(*f, c)

class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, output_size=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        self.output_size = output_size or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)
