import torch
import numpy as np
import time


class Mesh:
    '''
    mesh graph obj contains pos edge_index and x
    x reprensent feature in face obj
    edge_index is the sparse matrix of adj matrix
    pos represent position matrix
    '''

    def __init__(self, vertexes, faces):
        self.vertexes = vertexes
        self.faces = faces.t()[:10, :]
        self.edge_index = self.nodes = None
        self.sorted_st_dict = self.sorted_ed_dict = None
        self.sorted_st = self.sorted_ed = None
        # cal time
        start_time = time.time()
        self.create_graph()
        end_time = time.time()
        print('take {} s time for translate'.format(end_time-start_time))

    def create_graph(self):
        # conat center ox oy oz norm
        pos = self.vertexes[self.faces]
        point_x, point_y, point_z = pos[:, 0, :], pos[:, 1, :], pos[:, 2, :]
        centers = get_inner_center_vec(
            point_x, point_y, point_z
        )
        ox, oy, oz = get_three_vec(centers, point_x, point_y, point_z)
        norm = get_unit_norm_vec(point_x, point_y, point_z)
        self.nodes = torch.cat((centers, ox, oy, oz, norm), dim=1)
        # init neighbor dict to reduce time to be o(nlogn)
        self.init_neighbor_dict(self.faces)

    def init_neighbor_dict(self, faces):
        # concat  point_a point_b index
        # and get edge
        index = torch.arange(faces.size(0))
        edge_A = torch.cat((faces[:, :2], index.view(-1, 1)), dim=1)
        edge_B = torch.cat((faces[:, 1:], index.view(-1, 1)), dim=1)
        edge_C = torch.cat((faces[:, ::2], index.view(-1, 1)), dim=1)
        edge_num = torch.cat((edge_A, edge_B, edge_C), dim=0)
        # sort start_point and end_point
        start_point = edge_num[:, 0]
        end_point = edge_num[:, 1]
        # sort the index
        sorted_a, indices_a = torch.sort(start_point)
        sorted_b, indices_b = torch.sort(end_point)

        sorted_st = edge_num[indices_a, :]
        sorted_ed = edge_num[indices_b, :]
        print(sorted_st)

        # add indices dict for edge select
        sorted_index = torch.arange(faces.size(0)).view(-1, 1)
        sorted_st_with_index = torch.cat((sorted_st, sorted_index), dim=1)

        # sorted_st_unique, sorted_st_index = torch.unique(
        #     sorted_st_with_index, return_inverse=True)
        # sorted_st_index = sorted_st_index[:sorted_st_unique.size(0)]

        print(sorted_index)
        print(sorted_st_with_index)


def get_inner_center_vec(v1, v2, v3):
    '''
    v1 v2 v3 represent 3 vertexes of triangle
    v1 (n,3)
    '''
    a = get_distance_vec(v2, v3)
    b = get_distance_vec(v3, v1)
    c = get_distance_vec(v1, v2)
    x = torch.stack((v1[:, 0], v2[:, 0], v3[:, 0]), dim=1)
    y = torch.stack((v1[:, 1], v2[:, 1], v3[:, 1]), dim=1)
    z = torch.stack((v1[:, 2], v2[:, 2], v3[:, 2]), dim=1)
    dis = torch.stack((a, b, c), dim=1)
    return torch.stack((
        torch.sum((x * dis) / (a+b+c).repeat(3, 1).t(), dim=1),
        torch.sum((y * dis) / (a+b+c).repeat(3, 1).t(), dim=1),
        torch.sum((z * dis) / (a+b+c).repeat(3, 1).t(), dim=1),
    ), dim=1)


def get_distance_vec(v1, v2):
    '''
    get distance between
    vecter_1 and vecter_2
    v1 : (x,y,z)
    '''
    return torch.sqrt(torch.sum((v1-v2)**2, dim=1))


def get_unit_norm_vec(v1, v2, v3):
    '''
    xy X xz
    （y1z2-y2z1,x2z1-x1z2,x1y2-x2y1）
    '''
    xy = v2-v1
    xz = v3-v1
    x1, y1, z1 = xy[:, 0], xy[:, 1], xy[:, 2]
    x2, y2, z2 = xz[:, 0], xz[:, 1], xz[:, 2]
    norm = torch.stack((y1*z2-y2*z1, x2*z1-x1*z2, x1*y2-x2*y1), dim=1)
    vec_len = torch.sqrt(torch.sum(norm, dim=1))
    return norm / vec_len.repeat(3, 1).t()


def get_three_vec(center, v1, v2, v3):
    '''
    return ox oy oz vector
    '''
    return v1-center, v2-center, v3-center
