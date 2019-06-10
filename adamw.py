import torch.optim as optim

class AdamW(optim.Adam):
    def __init__(self, params, **kwargs):
        self.weight_decay = kwargs.pop('weight_decay')
        kwargs['weight_decay'] = 0
        super().__init__(params, **kwargs)

    def step(self):
        for group in self.param_groups:
            for param in group['params']:
                param.data.mul_(1 - self.weight_decay*group['lr'])
        super().step()
