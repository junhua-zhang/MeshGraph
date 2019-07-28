import torch
import torch.nn as nn


class FaceVectorConv(nn.Module):
    def __init__(self, output_channel=64):
        super(FaceVectorConv, self).__init__()
        self.rotate_mlp = nn.Sequential(
            nn.Conv1d(6, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(32, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, output_channel, 1),
            nn.BatchNorm1d(output_channel),
            nn.ReLU()
        )

    def forward(self, x):
        '''
        center ox oy oz norm
        3       3  3  3   3
        '''
        face_line = (
            self.rotate_mlp(x[:, 3:9]) +
            self.rotate_mlp(x[:, 6:12]) +
            self.rotate_mlp(torch.cat([x[:, 3:6], x[:, 9:12]], dim=1))
        ) / 3
        return self.fusion_mlp(face_line)


class PointConv(nn.Module):
    def __init__(self, output_channel=64):
        super(PointConv, self).__init__()

        self.spatial_mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, output_channel, 1),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        '''
        center ox oy oz norm
        '''
        return self.spatial_mlp(x[:, :3])


class MeshMlp(nn.Module):
    def __init__(self, output_channel=256):
        super(MeshMlp, self).__init__()
        self.pc = PointConv()
        self.fvc = FaceVectorConv()
        self.mlp = nn.Sequential(
            nn.Conv1d(131, output_channel, 1),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(),
            nn.Conv1d(output_channel, output_channel, 1),
            nn.BatchNorm1d(output_channel),
            nn.ReLU()
        )

    def forward(self, x):
        point_feature = self.pc(x)  # n 64
        face_feature = self.fvc(x)  # n 64
        fusion_feature = torch.cat(
            [x[:, 12:], point_feature, face_feature]  # n 64+64+3 = 131
            , dim=1
        )
        return self.mlp(fusion_feature)
