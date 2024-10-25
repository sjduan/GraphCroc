import sys
from os.path import dirname
sys.path.append(dirname(__name__)+'..')

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

import torch_geometric


from torch_geometric.utils import to_dense_batch, to_dense_adj, degree, add_remaining_self_loops, remove_self_loops

import logging
import time

from GraphCroc.UNET import Unet

from sklearn.metrics import f1_score, roc_auc_score

# dataset 
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

def train(model, train_loader: DataLoader, optimizer, scheduler, device, epoch):
    global writer
    model.train()

    total_loss = 0
    for data in train_loader:
        data.x = gen_node_feature(data)
        data = data.to(device)
        
        data.edge_index = add_remaining_self_loops(data.edge_index)[0]
        optimizer.zero_grad()
        label = data.y
        
        hs, mask = to_dense_batch(data.x, data.batch)
        hs = [hs[i][mask[i]] for i in range(len(hs))]
        gs = to_dense_adj(data.edge_index, data.batch)
        gs = [gs[i][:len(hs[i]), :len(hs[i])] for i in range(len(gs))]
        loss, _ = model(gs, hs)

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    scheduler.step()

def calculate_metrics(true_labels, predicted_labels):
    f1 = f1_score(true_labels, predicted_labels)
    ra = roc_auc_score(true_labels, predicted_labels)
    return f1, ra

def evaluate(model, train_loader, device):
    model.eval()

    trueLabel = []
    predLabel = []
    with torch.no_grad():
        for data in train_loader:
            data.x = gen_node_feature(data)
            data = data.to(device)
            
            data.edge_index = add_remaining_self_loops(data.edge_index)[0]
            hs, mask = to_dense_batch(data.x, data.batch)
            hs = [hs[i][mask[i]] for i in range(len(hs))]
            gs = to_dense_adj(data.edge_index, data.batch)
            gs = [gs[i][:len(hs[i]), :len(hs[i])] for i in range(len(gs))]
            _, o_gs = model(gs, hs)

            for og, g in zip(o_gs, gs):
                # og = (torch.sign(og-0.5)+1)/2
                trueLabel += g.int().cpu().numpy().flatten().tolist()
                predLabel += og.cpu().numpy().flatten().tolist()
    trueLabel = np.array(trueLabel)
    predLabel = np.array(predLabel)
    return (trueLabel==predLabel).sum()/trueLabel.size, *calculate_metrics(trueLabel, predLabel)

class ARG():
    pass

args = ARG()
args.input_dim = FEAT_DIM
args.act = 'GELU'
args.dim = 128
args.drop_p = 0
args.mask_ratio = 0.5
args.ks = [0.9, 0.8, 0.7, 0.6, 0.5] + [0]


num_epoch = 200

model = Unet(in_dim= args.input_dim, args=args).to(device)
optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch)

t = time.asctime()
t_str = t[20:]+t[4:7]+t[8:10]+'_'+t[11:13]+t[14:16]+t[17:19]
logging.basicConfig(filename='resLog/result_'+t_str+'.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

train_acc = []
test_acc = []
train_acc.append(evaluate(model, train_loader, device))
test_acc.append(evaluate(model, test_loader, device))
logging.info(f'Epoch [0/{num_epoch}], acc_train: {train_acc[-1][0]:.6f}, acc_test: {test_acc[-1][0]:.6f}, f1_train: {train_acc[-1][1]:.6f}, f1_test: {test_acc[-1][1]:.6f}, roc_auc_test: {test_acc[-1][2]:.6f}')
for epoch in range(num_epoch):
    train(model, train_loader, optimizer, scheduler, device, epoch)
    train_acc.append(evaluate(model, train_loader, device))
    test_acc.append(evaluate(model, test_loader, device))
    logging.info(f'Epoch [{epoch+1}/{num_epoch}], acc_train: {train_acc[-1][0]:.6f}, acc_test: {test_acc[-1][0]:.6f}, f1_train: {train_acc[-1][1]:.6f}, f1_test: {test_acc[-1][1]:.6f}, roc_auc_test: {test_acc[-1][2]:.6f}')

torch.save(model.state_dict(), 'tdModels/ae_model_degreefeature.pth')