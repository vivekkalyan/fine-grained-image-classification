import atexit
import os

import torch
from tqdm import tqdm

from logger import Logger
from lr_finder import LRFinder
from utils import get_git_hash, device, copy_runpy
from metrics import accuracy, unwrap_input
from onecycle import OneCycle

class Trainer:
    def __init__(self, model, criterion, optimizer, train_loader,
                 val_loader=None, name="experiment", experiments_dir="runs",
                 save_dir=None):
        self.device = device()
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self._epoch_count = 0
        self._best_loss = None
        self._best_acc = None
        if save_dir is None:
            save_dir = f"{self.get_num_dir(experiments_dir):04d}-{get_git_hash()}-{name}"
        self._save_dir = os.path.join(experiments_dir, save_dir)
        self.writer = Logger(self._save_dir)
        atexit.register(self.cleanup)

    def train(self, epochs=1):
        for epoch in range(epochs):
            self._epoch_count += 1
            print("\n----- epoch ", self._epoch_count, " -----")
            train_loss, train_acc = self._train_epoch()
            if self.val_loader:
                val_loss, val_acc = self._validate_epoch()
                if self._best_loss is None or val_loss < self._best_loss:
                    self.save_checkpoint('best_model')
                    self._best_loss = val_loss
                    print("new best val loss!")

    def train_one_cycle(self, epochs=1):
        self.onecycle = OneCycle(len(self.train_loader) * epochs,
                self.optimizer.defaults['lr'])
        self.train(epochs)
        self.onecycle = None

    def _train_epoch(self, save_histogram=False):
        self.model.train()
        running_loss = 0
        running_acc = 0
        for iter, (inputs, targets) in enumerate(tqdm(self.train_loader)):
            inputs = inputs.to(device())
            targets = targets.to(device())
            if self.onecycle is not None:
                lr, mom = next(self.onecycle)
                self.update_lr(lr)
                self.update_mom(mom)
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                batch_loss = self.criterion(outputs, targets)
                batch_acc = accuracy(outputs, targets)
                batch_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            running_loss += batch_loss.item()
            running_acc += batch_acc.item()
            if self.log_every(iter):
                self.writer.add_scalars("loss", {"train_loss":running_loss/float(iter+1)},
                                  (self._epoch_count - 1)*len(self.train_loader) + iter)
                self.writer.add_scalars("acc", {"train_acc":running_acc/float(iter+1)},
                                  (self._epoch_count - 1)*len(self.train_loader) + iter)
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = running_acc / len(self.train_loader)
        print(f"train loss: {epoch_loss:.5f} train acc: {epoch_acc:.5f}")
        return epoch_loss, epoch_acc

    def _validate_epoch(self):
        self.model.eval()
        running_loss = 0
        running_acc = 0
        for iter, (inputs, targets) in enumerate(tqdm(self.val_loader)):
            inputs = inputs.to(device())
            targets = targets.to(device())
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                batch_loss = self.criterion(outputs, targets)
                batch_acc = accuracy(outputs, targets)
            running_loss += batch_loss.item()
            running_acc += batch_acc.item()
            if self.log_every(iter):
                self.writer.add_scalars("loss", {"val_loss":running_loss/float(iter+1)},
                                  (self._epoch_count - 1)*len(self.val_loader) + iter)
                self.writer.add_scalars("acc", {"val_acc":running_acc/float(iter+1)},
                                  (self._epoch_count - 1)*len(self.val_loader) + iter)
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = running_acc / len(self.val_loader)
        print(f"val loss: {epoch_loss:.5f} val acc: {epoch_acc:.5f}")
        return epoch_loss, epoch_acc

    def get_num_dir(self, path):
        num_dir = len(os.listdir(path))
        return num_dir

    def save_checkpoint(self, fname):
        path = os.path.join(self._save_dir, fname)
        torch.save(dict(
            epoch=self._epoch_count,
            best_loss=self._best_loss,
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
        ), path)

    def load_checkpoint(self, fname):
        path = os.path.join(self._save_dir, fname)
        checkpoint = torch.load(path, map_location=lambda storage,
                loc: storage)
        self._epoch_count = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def log_every(self, i):
        return (i % 100)==0

    def update_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def update_mom(self, mom):
        keys = self.optimizer.param_groups[0].keys()
        for g in self.optimizer.param_groups:
            if 'momentum' in g.keys():
                g['momentum'] = mom
            elif 'betas' in g.keys():
                g['betas'] = mom if isinstance(mom, tuple) else (mom, g['betas'][1])
            else:
                raise ValueError

    def change_lr(self, lr):
        self.optimizer.defaults['lr'] = lr
        self.update_lr(lr)

    def find_lr(self, start_lr=1e-7, end_lr=100, num_iter=100):
        optimizer_state = self.optimizer.state_dict()
        self.change_lr(start_lr)
        self.lr_finder = LRFinder(self.model, self.optimizer, self.criterion,
                                  self.device)
        self.lr_finder.range_test(self.train_loader, end_lr=end_lr,
                                  num_iter=num_iter)
        self.optimizer.load_state_dict(optimizer_state)
        self.lr_finder.plot()

    def cleanup(self):
        copy_runpy(self._save_dir)
        path = os.path.join(self._save_dir, "./all_scalars.json")
        self.writer.export_scalars_to_json(path)
        self.writer.close()
