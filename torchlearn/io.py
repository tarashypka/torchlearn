from pathlib import Path

from pysimple.io import load_pickle, dump_pickle


class Savable:
    """Interface for objects that may be saved"""

    def __save__(self, filepath: Path):
        dump_pickle(filepath=filepath, obj=self)


class Loadable:
    """Interface for objects that may be loaded"""

    @classmethod
    def __load__(cls, filepath: Path):
        return load_pickle(filepath=filepath)
