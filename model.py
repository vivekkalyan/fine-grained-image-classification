import torchvision
import torch.nn as nn

from utils import device

class Resnet34(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        model = torchvision.models.resnet34(pretrained=True)
        in_features = model.fc.in_features
        self.resnet = nn.Sequential(*(list(model.children())[:-1]))
        self.fc = LinearLayer(in_features, out_features)

    def forward(self, inputs):
        x = inputs['image'].to(device())
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

