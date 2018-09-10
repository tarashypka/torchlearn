import os
from datetime import datetime as dt
from pathlib import Path
from typing import *

import dill as pickle


def report(*args, **kwargs):
    print(f'[{dt.now().strftime("%Y-%m-%s %H:%M:%S")}]', *args, **kwargs)


def plain_path(path: os.PathLike) -> Path:
    return Path(path).expanduser().absolute()


def ensure_dir(dirpath: os.PathLike) -> os.PathLike:
    dirpath = plain_path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath


def ensure_filedir(filepath: os.PathLike) -> os.PathLike:
    filepath = plain_path(filepath)
    ensure_dir(filepath.parent)
    return filepath


def dump_pickle(filepath: os.PathLike, obj: Any):
    filepath = ensure_filedir(filepath)
    with open(file=filepath, mode='wb') as f:
        pickle.dump(obj=obj, file=f)


def load_pickle(filepath: os.PathLike) -> Any:
    with open(file=filepath, mode='rb') as f:
        return pickle.load(file=f)


def read_lines(filepath: os.PathLike, **kwargs) -> Generator[str, None, None]:
    filepath = plain_path(filepath)
    kwargs.setdefault('mode', 'r')
    kwargs.setdefault('encoding', 'utf-8')
    with open(filepath, **kwargs) as f:
        for line in f:
            yield line.rstrip()


def write_lines(filepath: os.PathLike, lines: List[str], **kwargs) -> None:
    filepath = ensure_filedir(filepath)
    kwargs.setdefault('mode', 'w')
    kwargs.setdefault('encoding', 'utf-8')
    with open(filepath, **kwargs) as f:
        for line in lines:
            f.write(line + '\n')
