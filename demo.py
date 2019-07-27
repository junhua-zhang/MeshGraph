import torch
# from torch_geometric.data import Data

# edge_index = torch.tensor([[0, 1, 1, 2],
#                            [1, 0, 2, 1]], dtype=torch.long)
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

# data = Data(x=x, edge_index=edge_index)
# # print(data)

# # device = torch.device('cuda')
# # data.to(device)
# print(data)

# from torch_geometric.datasets import TUDataset
# from torch_geometric.data import DataLoader
# from torch_scatter import scatter_mean
# dataset = TUDataset(root='datasets/ENZYMES', name='ENZYMES')
# loader = DataLoader(dataset, batch_size=32, shuffle=True)
# # print(dataset)

# # print(len(dataset))
# # print(dataset.num_classes)
# # print(dataset.num_node_features)

# # data = dataset[0]
# # print(data)
# # print(data.is_undirected())
# # dataset.shuffle()
# for batch in loader:
#     # print(batch)
#     x = scatter_mean(batch.x, batch.batch, dim=0)
#     print(x.size())

# import torch_geometric.transforms as T
# from torch_geometric.datasets import ShapeNet
# # dataset = ShapeNet(root='datasets/shapenet', categories=['Airplane'],
# #                    pre_transform=T.KNNGraph(k=6))
# dataset = ShapeNet(root='datasets/shapenet', categories=['Airplane'],
#                     pre_transform=T.KNNGraph(k=6),
#                     transform=T.RandomTranslate(0.01))
# print(dataset[0])
# from torch_geometric.nn import GCNConv
# import torch.nn.functional as F
# import torch
# from torch_geometric.datasets import Planetoid

# dataset = Planetoid(root='datasets/Cora', name='Cora')


# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = GCNConv(dataset.num_node_features, 16)
#         self.conv2 = GCNConv(16, dataset.num_classes)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=1)


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net().to(device)
# data = dataset[0].to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
# model.train()
# for epoch in range(200):
#     optimizer.zero_grad()
#     out = model(data)
#     loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
#     loss.backward()
#     optimizer.step()


# model.eval()
# _, pred = model(data).max(dim=1)
# correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
# acc = correct / data.test_mask.sum().item()
# print('Accuracy:{:.4f}'.format(acc))

# str_1 = '0 ,1 ,2'
# ids = str_1.split(',')
# print(ids)


# def remove_isolated_vertices(data):
#     mask = torch.zeros(data['num_nodes'], dtype=torch.uint8)
#     mask[data['edge_index'].flatten()] = 1
#     assoc = torch.full((data['num_nodes'], ), -1, dtype=torch.long)
#     assoc[mask] = torch.arange(mask.sum())
#     print(assoc)
#     new_edge_index = assoc[data['edge_index']]
#     print(new_edge_index)
#     new_pos = data['pos'][mask]
#     return new_pos, new_edge_index


# data = {}

# x = torch.tensor(
#     [
#         [-1, 0, 1],
#         [0, 1, 1],
#         [1, 0, 1],
#         [0, -1, 1],
#         [3, 0, 1]
#     ]
# )
# data['x'] = x
# data['pos'] = x
# edge_index = torch.tensor(
#     [
#         [0, 0, 0, 1, 1, 2, 2, 2, 3, 3],
#         [1, 2, 3, 0, 2, 0, 1, 3, 0, 2]
#     ]
# )

# data['edge_index'] = edge_index
# data['num_nodes'] = 5

# a, b = remove_isolated_vertices(data)

# # print(a)
# # print(b)

# p = torch.tensor([0,1,2,3,-1])

# mask = torch.tensor([
#     [0,1,2,3],
#     [1,2,3,4]
# ])

# print(p[mask])

# b = torch.tensor([0,0,1,2])
# a = torch.tensor([
#     [0, 0],
#     [0, 1],
#     [1, 2],
#     [2, 3]
# ])


# c = torch.argsort(b)
# print(c)


# s = (1,2)
# print(s)
# print(type(s))
# b = set(1,2)

a = [1, 2, 3]
print(a[2:3])
