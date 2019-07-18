from .base_options import base_options


class train_options(base_options):
    def initialize(self):
        base_options.initialize(self)
        self.parser.add_argument(
            '--continue_train', action='store_true', help='continue training: load the latest model'
        )
        self.parser.add_argument(
            '--last_epoch', default='latest', help='which epoch to load?'
        )
        # optimizer param
        self.parser.add_argument(
            '--lr', default=0.01, type=float, help='learning rate'
        )
        self.parser.add_argument(
            '--lr_policy', default='step', type=str, help='learning rate policy: step|'
        )
        self.parser.add_argument(
            '--milestones', default='30,60', help='milestones for MultiStepLR'
        )
        self.is_train = True
