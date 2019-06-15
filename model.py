import torch
import torch.nn as nn
import torchvision

from utils import device

from functools import partial
models = {
    "resnet18": torchvision.models.resnet18,
    "resnet34": torchvision.models.resnet34,
    "resnet50": torchvision.models.resnet50,
    "resnet101": torchvision.models.resnet101,
    "resnet152": torchvision.models.resnet152,
}

class Resnet(nn.Module):
    def __init__(self, model_name, out_features):
        super().__init__()
        model = models[model_name](pretrained=True)
        in_features = model.fc.in_features
        self.resnet = nn.Sequential(*(list(model.children())[:-2]))
        concat_pool = AdaptiveConcatPool2d()
        flatten = Flatten()
        fc1 = LinearLayer(in_features*2, 512, dropout=0.25,
                activation=nn.ReLU())
        fc2 = LinearLayer(512, out_features, dropout=0.5)
        self.fc = nn.Sequential(concat_pool, flatten, fc1, fc2)
        freeze(self.resnet)
        apply_init(self.fc, nn.init.kaiming_normal_)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x

    def unfreeze(self):
        for param in self.resnet.parameters():
            param.requires_grad = True

class Flatten(nn.Module):
    'Flatten to a single dimension'
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)


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

BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

def freeze(module, freeze_bn=False):
    apply_leaf(module, partial(cond_freeze, freeze_bn=freeze_bn))

def cond_freeze(module, freeze_bn):
    "Don't freeze batchnorm layers unless `freeze_bn` is True"
    if isinstance(module, BN_TYPES) and not freeze_bn: requires_grad(module, True)
    else: requires_grad(module, False)

def init_default(module, func=nn.init.kaiming_normal_):
    "Initialize `module` weights with `func` and set `bias` to 0."
    if func:
        if hasattr(module, 'weight'): func(module.weight)
        if hasattr(module, 'bias') and hasattr(module.bias, 'data'): module.bias.data.fill_(0.)
    return module

def cond_init(module, init_func):
    "Initialize the non-batchnorm layers of `module` with `init_func`."
    if (not isinstance(module, BN_TYPES)) and requires_grad(module): init_default(module, init_func)

def apply_init(module, init_func):
    "Initialize all non-batchnorm layers of `module` with `init_func`."
    apply_leaf(module, partial(cond_init, init_func=init_func))

def apply_leaf(module, func):
    "Apply `func` to children of `module`."
    c = children(module)
    if isinstance(module, nn.Module): func(module)
    for l in c: apply_leaf(l,func)

def children(module):
    "Get children of `module`."
    return list(module.children())

def requires_grad(module, bool=None):
    "If `bool` is not set return `requires_grad` of first param, else set `requires_grad` on all params as `bool`"
    params = list(module.parameters())
    if not params: return None
    if bool is None: return params[0].requires_grad
    for p in params: p.requires_grad=bool
