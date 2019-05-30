import torch.nn as nn
import torch

class LossInputs(nn.Module):
    def __init__(self, criterion):
        super(LossInputs, self).__init__()
        self.criterion = criterion

    def forward(self, outputs, inputs):
        targets = inputs['class_id']
        return self.criterion(outputs, targets)


