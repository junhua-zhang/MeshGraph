import torch
import numpy as np


class Mesh:
    '''
    mesh graph obj contains pos edge_index and x
    x reprensent feature in face obj
    edge_index is the sparse matrix of adj matrix
    pos represent position matrix
    '''

    def __init__(self, vertexes, faces):
        self.vertexes = vertexes
        self.faces = faces.t()
        self.edge_index = self.nodes = None

        # find point and face indices
        self.sorted_point_to_face = None
        self.sorted_point_to_face_index_dict = None
        # create graph
        self.create_graph()

    def create_graph(self):
        # conat center ox oy oz norm
        pos = self.vertexes[self.faces]
        point_x, point_y, point_z = pos[:, 0, :], pos[:, 1, :], pos[:, 2, :]
        centers = get_inner_center_vec(
            point_x, point_y, point_z
        )
        ox, oy, oz = get_three_vec(centers, point_x, point_y, point_z)
        norm = get_unit_norm_vec(point_x, point_y, point_z)
        # cat the vecter
        self.nodes = torch.cat((centers, ox, oy, oz, norm), dim=1)
        print('get_edge_index')
        # init neighbor dict to reduce time to be o(nlogn)
        self.init_neighbor_dict(self.faces)
        print('init_neighbor_dict_done')
        self.get_edge_index(self.faces)
        print('get edge_index_done')

    def init_neighbor_dict(self, faces):
        # concat  point_a point_b index
        # and get edge
        index = torch.arange(faces.size(0))
        edge_A = torch.cat((faces[:, :2], index.view(-1, 1)), dim=1)
        edge_B = torch.cat((faces[:, 1:], index.view(-1, 1)), dim=1)
        edge_C = torch.cat((faces[:, ::2], index.view(-1, 1)), dim=1)
        edge_num = torch.cat((edge_A, edge_B, edge_C), dim=0)
        # sort start_point and end_point
        start_point_with_face = edge_num[:, ::2]
        end_point_with_face = edge_num[:, 1:]
        point_with_face = torch.cat(
            (start_point_with_face, end_point_with_face),
            dim=0
        )
        # sort the point and faces
        _, sorted_point_index = torch.sort(point_with_face[:, 0])
        sorted_point = point_with_face[sorted_point_index, :]
        self.sorted_point_to_face = sorted_point
        self.sorted_point_to_face_index_dict = get_indices_dict(
            sorted_point[:, 0])

    def get_edge_index(self, faces):
        edge_index = torch.tensor([])
        for index, value in enumerate(faces):
            total_num = 0
            # index_tensor = torch.tensor(index).repeat(1, 1)
            v1, v2, v3 = value
            v_indeces = self.sorted_point_to_face_index_dict[
                v1.item()
            ]
            v_face_index_vec = self.sorted_point_to_face[v_indeces, 1]
            size = v_face_index_vec.size(0)
            index_tensor = torch.tensor(index).repeat(size)
            temp_edge = torch.stack(
                (index_tensor, v_face_index_vec),
                dim=1
            )
            edge_index = torch.cat(
                (edge_index, temp_edge), dim=0) if edge_index.size(0) > 0 else temp_edge

            v_indeces = self.sorted_point_to_face_index_dict[
                v2.item()
            ]
            v_face_index_vec = self.sorted_point_to_face[v_indeces, 1]
            size = v_face_index_vec.size(0)
            index_tensor = torch.tensor(index).repeat(size)
            temp_edge = torch.stack(
                (index_tensor, v_face_index_vec),
                dim=1
            )
            edge_index = torch.cat(
                (edge_index, temp_edge), dim=0) if edge_index.size(0) > 0 else temp_edge

            v_indeces = self.sorted_point_to_face_index_dict[
                v3.item()
            ]
            v_face_index_vec = self.sorted_point_to_face[v_indeces, 1]
            size = v_face_index_vec.size(0)
            index_tensor = torch.tensor(index).repeat(size)
            temp_edge = torch.stack(
                (index_tensor, v_face_index_vec),
                dim=1
            )
            edge_index = torch.cat(
                (edge_index, temp_edge), dim=0) if edge_index.size(0) > 0 else temp_edge
        self.edge_index = edge_index


def get_indices_dict(vec):
    start = 0
    cur_value = vec[0]
    indices_dict = {}
    for index, value in enumerate(vec):
        if value == cur_value:
            continue
        else:
            indices_dict[cur_value.item()] = torch.arange(start, index)
            cur_value = value
            start = index
    indices_dict[cur_value.item()] = torch.arange(start, len(vec))
    return indices_dict


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
