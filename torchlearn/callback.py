from datetime import datetime as dt
from pathlib import Path
from time import time
from typing import *

import torch
from torch import nn

from pysimple.io import dump_pickle, ensure_dir
from torchlearn.io import Savable


class Callback:
    pass


class SaveCallback(Callback):

    def __init__(self, obj: Any, name: str, save_per_min: int, save_dir: Path):
        self.obj = obj
        self.name = name
        self.save_per_min = save_per_min
        self.save_dir = ensure_dir(dirpath=save_dir)

        self.prev_save_: int = None

    def __call__(self, *args, **kwargs):
        now = time()
        if self.prev_save_ is None or now - self.prev_save_ > self.save_per_min * 60:
            save_path = self.save_dir / f'{self.name}_{dt.now().strftime("%y%m%d%H%M%S")}.bin'
            print('Save', self.name, 'into', save_path, '...')
            if isinstance(self.obj, Savable):
                self.obj.save(filepath=save_path)
            elif isinstance(self.obj, nn.Module):
                torch.save(obj=self.obj, f=save_path)
            else:
                dump_pickle(filepath=save_path, obj=self.obj)
            self.prev_save_ = now
