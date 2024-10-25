import sys
from os.path import dirname
sys.path.append(dirname(__name__)+'..')

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

import torch_geometric


from torch_geometric.utils import to_dense_batch, to_dense_adj, degree, add_remaining_self_loops

from GraphCroc.UNET_onebranch import Unet

from wl_test_util import wl_algorithm

dataset = torch_geometric.datasets.TUDataset('../data/', 'IMDB-BINARY')

idx = torch.randperm(len(dataset))
cut_off = int(len(dataset) * 0.8)
train_dataset = dataset[idx[:cut_off]]
test_dataset = dataset[idx[cut_off:]]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# dataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(f'training size {len(train_dataset)}, testing set {len(test_dataset)}, Number of graphs per batch: {next(iter(train_loader)).y.shape[0]}')


feature_dim = 0
for g in dataset:
    deg = degree(g.edge_index[0], g.num_nodes)
    feature_dim = max(feature_dim, deg.max().item())
print(f'feature_dim: {feature_dim}')
FEAT_DIM = int(feature_dim)+1 if feature_dim < 400 else 400

def gen_node_feature(data):
    deg = degree(data.edge_index[0], data.num_nodes).long()
    deg[deg >= FEAT_DIM] = FEAT_DIM - 1
    feat = F.one_hot(deg, num_classes=FEAT_DIM).float()
    return feat

def evaluate(model, train_loader, device):
    model.eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for data in train_loader:
            # data.x = torch.ones((data.num_nodes,1))
            data.x = gen_node_feature(data)
            data = data.to(device)
            
            data.edge_index = add_remaining_self_loops(data.edge_index)[0]
            hs, mask = to_dense_batch(data.x, data.batch)
            hs = [hs[i][mask[i]] for i in range(len(hs))]
            gs = to_dense_adj(data.edge_index, data.batch)
            gs = [gs[i][:len(hs[i]), :len(hs[i])] for i in range(len(gs))]
            _, o_gs = model(gs, hs)

            for og, g in zip(o_gs, gs):
                og = (torch.sign(og-0.5)+1)/2
                result = wl_algorithm(g, og, iterations=3)
                total += 1
                correct += result
    return correct/total


class ARG():
    pass

args = ARG()
args.input_dim = FEAT_DIM
args.act = 'GELU'
args.dim = 32
args.drop_p = 0
args.mask_ratio = 0.5
args.ks = [0.9, 0.8, 0.7, 0.6, 0.5] + [0]


num_epoch = 200

model = Unet(in_dim= 1, args=args).to(device)
model.load_state_dict(torch.load('tdModels/ae_model_degreefeature.pth'))
# model.load_state_dict(torch.load('tdModels/ae_model_selfcor.pth'))

print('WL-test: ', evaluate(model, test_loader, device))
