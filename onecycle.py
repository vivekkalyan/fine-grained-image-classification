import numpy as np

def annealing_cos(start, end, pct):
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start-end) * cos_out/2

class OneCycle():
    def __init__(self, num_iter, lr_max, mom_min=0.85, mom_max=0.95,
            div_factor=25., peak_pct=0.3):
        self.lr_max = lr_max
        self.lr_low = lr_max/div_factor
        self.lr_min = lr_max/(div_factor*1000)
        self.mom_min = mom_min
        self.mom_max = mom_max
        self.num_iter = num_iter
        self.peak_iter = num_iter*peak_pct
        self.lrs = []
        self.moms = []
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.num_iter: self.i = 0
        self.i += 1
        lr = self.calc_lr(self.i)
        mom = self.calc_mom(self.i)
        return (lr, mom)

    def calc_lr(self, i):
        if i < self.peak_iter:
            lr = annealing_cos(self.lr_low, self.lr_max,
                    i/self.peak_iter)
        else:
            lr = annealing_cos(self.lr_max, self.lr_min,
                    (i - self.peak_iter)/(self.num_iter - self.peak_iter))
        self.lrs.append(lr)
        return lr

    def calc_mom(self, i):
        if i < self.peak_iter:
            mom = annealing_cos(self.mom_max, self.mom_min,
                    i/self.peak_iter)
        else:
            mom = annealing_cos(self.mom_min, self.mom_max,
                    (i - self.peak_iter)/(self.num_iter - self.peak_iter))
        self.moms.append(mom)
        return mom
