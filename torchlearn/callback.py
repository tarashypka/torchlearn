from datetime import datetime as dt
from pathlib import Path
from time import time
from typing import *

import torch
from torch import nn

from pysimple.io import dump_pickle
from torchlearn.io import Savable


class Callback:
    pass


class SaveCallback(Callback):

    def __init__(self, obj_to_save: Any, save_per_min: int, save_name: str):
        self.obj_to_save = obj_to_save
        self.save_per_min = save_per_min
        self.save_name = save_name

        self.prev_save_: int = None
        self.cache_dir_ = Path('.torchtext')
        self.cache_dir_.mkdir(exist_ok=True, parents=True)

    def __call__(self, *args, **kwargs):
        now = time()
        if self.prev_save_ is None or now - self.prev_save_ > self.save_per_min * 60:
            save_path = self.cache_dir_ / f'{self.save_name}_{dt.now().strftime("%Y%m%d%H%M%S")}.bin'
            print('Save', self.save_name, 'into', save_path, '...')
            if isinstance(self.obj_to_save, Savable):
                self.obj_to_save.__save__(filepath=save_path)
            elif isinstance(self.obj_to_save, nn.Module):
                torch.save(obj=self.obj_to_save, f=save_path)
            else:
                dump_pickle(filepath=save_path, obj=self.obj_to_save)
            self.prev_save_ = now
