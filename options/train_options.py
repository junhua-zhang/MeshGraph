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
            '--lr', default=1e-3, type=float, help='learning rate'
        )
        self.parser.add_argument(
            '--final_lr', default=0.1, type=float, help='final learning rate'
        )
        # self.parser.add_argument(
        #     '--momentum', default=0.9, type=float, help='momentum'
        # )
        # self.parser.add_argument(
        #     '--weight_decay', default=0.0005, type=float, help='weight decay'
        # )
        self.parser.add_argument(
            '--lr_policy', default='step', type=str, help='learning rate policy: step|'
        )
        self.parser.add_argument(
            '--milestones', default='30,60', help='milestones for MultiStepLR'
        )
        self.parser.add_argument(
            '--gamma', default=0.1, type=float, help='gamma for MultiStepLR'
        )

        # model
        self.is_train = True
