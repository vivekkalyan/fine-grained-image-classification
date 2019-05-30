import os
from tensorboardX import SummaryWriter

class Logger(SummaryWriter):
    def __init__(self, log_dir, **kwargs):
        self.dir_create = 0
        self.log_dir = log_dir
        super().__init__(log_dir, **kwargs)

    def _create_dir(self):
        os.mkdir(self.log_dir)
        self.dir_create += 1
        return self._get_file_writer()

    def _get_file_writer(self):
        if self.dir_create < 1: self.dir_create += 1
        elif self.dir_create == 1: return self._create_dir()
        else: return super()._get_file_writer()
