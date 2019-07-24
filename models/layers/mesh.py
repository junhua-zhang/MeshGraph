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
        self.edge_index = self.nodes = torch.tensor([])
        self.create_graph()

    def create_graph(self):
        self.edge_index.long()
        for i, face in enumerate(self.faces):
            print('{}th face processing..'.format(i))
            x, y, z = self.vertexes[face, :]
            # find related face and add to graph
            related_face = torch.tensor(find_relate(
                face, self.faces, i), dtype=torch.long).view(-1, 2)

            face = Face(x, y, z)
            # stack the tensor value

            if self.nodes.size(0):
                self.nodes = torch.cat(
                    (self.nodes, face.to_tensor_vecter()), dim=0)
            else:
                self.nodes = face.to_tensor_vecter()

            # add edge_index
            if self.edge_index.size(0):
                self.edge_index = torch.cat(
                    (self.edge_index, related_face), dim=0)
            else:
                self.edge_index = related_face
            
            if i > 20:
                break
        print(self.nodes)
        print(self.edge_index)


def find_relate(pos_a, pos_list, i):
    '''
    at least i relavent to itself
    '''
    result = []
    for index, pos_b in enumerate(pos_list):
        if is_contain(pos_a, pos_b):
            result.append([i, index])
    return result


def is_contain(a, b):
    '''
    return True if a contains b else false 
    such as [1,2,3]
            [2,4,5]
    will return True 
    '''
    return True if [x for x in a if x in b] else False


class Face:
    '''
    every face of each mesh.
    a mesh will contains many face
    x,y,z represent (x,y,z) index of point  
    '''

    def __init__(self, x, y, z):
        self.norm = None
        self.ox = self.oy = self.oz = None
        self.center = None

        self.initialize(x, y, z)

    def initialize(self, x, y, z):
        o = get_inner_center(x, y, z)
        self.center = torch.tensor([_ for _ in o])
        self.ox, self.oy, self.oz = get_three_vector(o, x, y, z)
        self.norm = torch.tensor([_ for _ in get_norm(x, y, z)])
        self.ox = torch.tensor([_ for _ in self.ox])
        self.oy = torch.tensor([_ for _ in self.oy])
        self.oz = torch.tensor([_ for _ in self.oz])

    def to_tensor_vecter(self):
        '''
        [center ox  oy  oz   norm ] 
            3   3   3   3    3
        '''
        return torch.cat((self.center, self.ox, self.oy, self.oz, self.norm), dim=0)


def get_inner_center(x, y, z):
    '''
    get inner point by x y z
    A(x1,y1),B(x2,y2),C(x3,y3),BC=a,CA=b,AB=c
    x (X1,Y1,Z1)
    y (X2,Y2,Z2)
    z (X3,Y3,Z3)
    （（aX1+bX2+cX3)/(a+b+c),（aY1+bY2+cY3)/(a+b+c) , （aZ1+bZ2+cZ3)/(a+b+c) ）
    '''
    x1, y1, z1 = x
    x2, y2, z2 = y
    x3, y3, z3 = z
    a = get_distance(y, z)
    b = get_distance(z, x)
    c = get_distance(x, y)
    return (a*x1+b*x2+c*x3)/(a+b+c), (a*y1+b*y2+c*y3)/(a+b+c), (a*z1+b*z2+c*z3)/(a+b+c)


def get_three_vector(o, x, y, z):
    '''
    return ox oy oz vector
    '''
    xo, yo, zo = o
    x1, y1, z1 = x
    x2, y2, z2 = y
    x3, y3, z3 = z
    return (x1-xo, y1-yo, z1-zo), (x2-xo, y2-yo, z2-zo), (x3-xo, y3-yo, z3-zo)


def get_norm(x, y, z):
    '''
    xy X xz 
    （y1z2-y2z1,x2z1-x1z2,x1y2-x2y1）
    '''
    x1, y1, z1 = x
    x2, y2, z2 = y
    x3, y3, z3 = z
    xy = (x2-x1, y2-y1, z2-z1)
    xz = (x3-x1, y3-y1, z3-z1)
    x1, y1, z1 = xy
    x2, y2, z2 = xz
    return y1*z2-y2*z1, x2*z1-x1*z2, x1*y2-x2*y1


def get_distance(a, b):
    '''
    get distance between a and b 
    a and b is (x,y,z)
    '''
    x1, y1, z1 = a
    x2, y2, z2 = b
    return np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
