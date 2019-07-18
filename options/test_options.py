from .base_options import base_options


class test_options(base_options):
    def initialize(self):
        base_options.initialize(self)
        self.is_train = False
