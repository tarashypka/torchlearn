from torchlearn.callback import Callback
from torchlearn.io import Savable, Loadable


class Trainer(Savable, Loadable):

    def __init__(self):
        self.callbacks = []

    def train(self, *args, **kwargs):
        pass

    def add_callback(self, callback: Callback):
        self.callbacks.append(callback)
