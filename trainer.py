import atexit
import os

import torch
from tqdm import tqdm

from logger import Logger
from lr_finder import LRFinder
from utils import get_git_hash, device, copy_runpy

class Trainer:
    def __init__(self, model, criterion, optimizer, train_loader,
                 val_loader=None, name="experiment", experiments_dir="runs"):
        self.device = device()
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self._epoch_count = 0
        self._best_loss = None
        save_dir = f"{self.get_num_dir(experiments_dir):04d}-{get_git_hash()}-{name}"
        self._save_dir = os.path.join(experiments_dir, save_dir)
        self.writer = Logger(self._save_dir)
        atexit.register(self.cleanup)

    def train(self, epochs=1):
        for epoch in range(epochs):
            self._epoch_count += 1
            print("\n----- epoch ", self._epoch_count, " -----")
            train_loss = self._train_epoch()
            if self.val_loader:
                val_loss = self._validate_epoch()
                if self._best_loss is None or val_loss < self._best_loss:
                    self._save_checkpoint('best_model')
                    self._best_loss = val_loss
                    print("new best val loss!")

    def _train_epoch(self, save_histogram=False):
        self.model.train()
        running_loss = 0
        for iter, inputs in enumerate(tqdm(self.train_loader)):
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                batch_loss = self.criterion(outputs, inputs)
                batch_loss.backward()
                self.optimizer.step()
            running_loss += batch_loss.item()
            if self.log_every(iter):
                self.writer.add_scalars("loss", {"train_loss":running_loss/float(iter+1)},
                                  (self._epoch_count - 1)*len(self.train_loader) + iter)
        epoch_loss = running_loss / len(self.train_loader)
        print(f"train loss: {epoch_loss:.5f}")
        return epoch_loss

    def _validate_epoch(self):
        self.model.eval()
        running_loss = 0
        for iter, inputs in enumerate(tqdm(self.val_loader)):
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                batch_loss = self.criterion(outputs, inputs)
            running_loss += batch_loss.item()
            if self.log_every(iter):
                self.writer.add_scalars("loss", {"val_loss":running_loss/float(iter+1)},
                                  (self._epoch_count - 1)*len(self.val_loader) + iter)
        epoch_loss = running_loss / len(self.val_loader)
        print(f"val loss: {epoch_loss:.5f}")
        return epoch_loss

    def get_num_dir(self, path):
        num_dir = len(os.listdir(path))
        return num_dir

    def _save_checkpoint(self, fname):
        path = os.path.join(self._save_dir, fname)
        torch.save(dict(
            epoch=self._epoch_count,
            best_loss=self._best_loss,
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            ), path)

    def log_every(self, i):
        return (i % 100)==0 and i!=0

    def update_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def update_mom(self, mom):
        for g in self.optimizer.param_groups:
            g['momentum'] = mom

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
