import random
import subprocess
import sys
from shutil import copyfile

import torch
import numpy as np

def get_git_hash():
    try:
        is_git_repo = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
    except FileNotFoundError:
        return ""

    if _decode_bytes(is_git_repo) == "true":
        git_hash = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  stdout=subprocess.PIPE).stdout
        return _decode_bytes(git_hash)
    else:
        return ""

def _decode_bytes(b: bytes):
    return b.decode("ascii")[:-1]

def copy_runpy(save_dir):
    filename = sys.argv[0]
    dest = save_dir+ '/' + filename
    copyfile(filename,dest)

def set_deterministic(seed=42):
    """
    Make experiments as reproducible as possible by setting a common
    seed
    Note: due to different implementations on CUDA and CPU, results
    will be different on CUDA and CPU
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def savePickle(objectToSave, savepath):
    print(f"Saving at {savepath}")
    save_file = open(savepath, 'wb')
    pickle.dump(objectToSave, save_file)

def loadPickle(loadpath):
    print(f"Loading from {loadpath}")
    opened_file = open(loadpath,'rb')
    return pickle.load(opened_file)

def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
