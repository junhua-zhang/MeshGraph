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
        xy = torch.unsqueeze(x[:, 3:9], dim=0).transpose(1, 2)
        yz = torch.unsqueeze(x[:, 6:12], dim=0).transpose(1, 2)
        xz = torch.unsqueeze(
            torch.cat([x[:, 3:6], x[:, 9:12]], dim=1), dim=0).transpose(1, 2)
        face_line = (
            self.rotate_mlp(xy) +
            self.rotate_mlp(yz) +
            self.rotate_mlp(xz)
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
        x = torch.unsqueeze(x[:, :3], dim=0).transpose(1, 2)
        # print(x.size())
        return self.spatial_mlp(x)


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
        self.global_pooling = nn.Sequential(
            nn.AdaptiveMaxPool1d(1024),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(64),
            nn.ReLU()

        )

    def forward(self, x):
        point_feature = self.pc(x)  # n 64
        face_feature = self.fvc(x)  # n 64
        norm = torch.unsqueeze(x[:, 12:], dim=0).transpose(1, 2)
        fusion_feature = torch.cat(
            [norm, point_feature, face_feature]  # n 64+64+3 = 131
            , dim=1
        )
        return self.global_pooling(self.mlp(fusion_feature))  # 1 64 m
