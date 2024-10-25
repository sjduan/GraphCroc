import sys
from os.path import dirname
sys.path.append(dirname(__name__)+'..')

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

import torch_geometric
import torch_geometric.transforms as T

from torch_geometric.utils import to_dense_batch, to_dense_adj, degree, add_remaining_self_loops, remove_self_loops


from GraphCroc.UNET import Encoder

from GraphUNET.ops import GCN, norm_g

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
FEAT_DIM = int(feature_dim)+1

class ARG():
    pass

class Classifier(nn.Module):
	def __init__(self, num_node, dim, hidden_dim, num_class, dp=0) -> None:
		super(Classifier, self).__init__()
		self.fc = nn.Linear(num_node * dim, hidden_dim)
		self.ln = nn.LayerNorm(hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
		self.ln2 = nn.LayerNorm(hidden_dim // 2)
		self.fc3 = nn.Linear(hidden_dim // 2, num_class)
		self.dp = nn.Dropout(dp)
	
	def forward(self, x):
		x = x.view(x.shape[0], -1)
		x = self.dp(x)
		x = self.ln(F.relu(self.fc(x)))
		x = self.dp(x)
		x = self.ln2(F.relu(self.fc2(x)))
		x = self.dp(x)
		x = self.fc3(x)
		return x

class clsEncoder(nn.Module):
	def __init__(self, in_dim, args) -> None:
		super(clsEncoder, self).__init__()
		self.act = getattr(nn, args.act)()
		self.s_gcn = GCN(in_dim, args.dim, self.act, args.drop_p)
		self.s_ln = nn.LayerNorm(args.dim)

		self.g_enc = Encoder(args.ks, args.dim, self.act, args.drop_p)

		self.bot_gcn = GCN(args.dim, args.dim, self.act, args.drop_p)
		self.bot_ln = nn.LayerNorm(args.dim)
	
	def forward(self, gs, hs):
		new_hs = []
		for g, h in zip(gs, hs):
			g = norm_g(g)
			h = self.s_gcn(g, h)
			h = self.s_ln(h)
			g, h = self.g_enc(g, h)
			g = norm_g(g)
			h = self.bot_gcn(g, h)
			h = self.bot_ln(h)
			new_hs.append(h)
		new_hs = torch.stack(new_hs)
		return new_hs

class GraphCLS(nn.Module):
	def __init__(self, enc_node, dim, hidden_dim, num_class, encoder, enc_freeze, dp=0) -> None:
		super(GraphCLS, self).__init__()
		self.encoder = encoder
		if enc_freeze:
			for param in self.encoder.parameters():
				param.requires_grad = False
		self.classifier = Classifier(enc_node, dim, hidden_dim, num_class, dp)
	
	def forward(self, gs, hs):
		hs = self.encoder(gs, hs)
		x = self.classifier(hs)
		return x
	
def gen_node_feature(data):
    deg = degree(data.edge_index[0], data.num_nodes).long()
    feat = F.one_hot(deg, num_classes=FEAT_DIM).float()
    return feat

def train(model, train_loader, optimizer, scheduler, device, epoch):
    model.train()

    total_loss = 0
    for data in train_loader:
        # data.x = torch.ones((data.num_nodes,1))
        data.x = gen_node_feature(data)
        data = data.to(device)
        data.edge_index = add_remaining_self_loops(data.edge_index)[0]
        optimizer.zero_grad()
        label = data.y
        # data.x = degree(data.edge_index[1], num_nodes=data.num_nodes).unsqueeze_(-1)
        hs, mask = to_dense_batch(data.x, data.batch)
        hs = [hs[i][mask[i]] for i in range(len(hs))]
        gs = to_dense_adj(data.edge_index, data.batch)
        gs = [gs[i][:len(hs[i]), :len(hs[i])] for i in range(len(gs))]
        outputs = model(gs, hs)

        loss = nn.CrossEntropyLoss()(outputs, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    scheduler.step()

@torch.no_grad()
def evaluate(model, train_loader, device):
    model.eval()

    total_loss = 0
    total = 0
    correct = 0
    for data in train_loader:
        data.x = gen_node_feature(data)
        data = data.to(device)
        total += data.y.size(0)
        data.edge_index = add_remaining_self_loops(data.edge_index)[0]
        label = data.y
        hs, mask = to_dense_batch(data.x, data.batch)
        hs = [hs[i][mask[i]] for i in range(len(hs))]
        gs = to_dense_adj(data.edge_index, data.batch)
        gs = [gs[i][:len(hs[i]), :len(hs[i])] for i in range(len(gs))]
        outputs = model(gs, hs)
        pred = outputs.argmax(dim=1)
        correct += pred.eq(label).sum().item()
    return correct / total # how many elements ratio is different

args = ARG()
args.act = 'GELU'
args.dim = 128
args.drop_p = 0
args.mask_ratio = 0.5
args.ks = [0.9, 0.8, 0.7, 0.6, 0.5] + [0]

encoder = clsEncoder(FEAT_DIM, args).to(device)
encoder.load_state_dict(torch.load('tdModels/ae_model_degreefeature.pth'), strict=False)

epochs = 100

model = GraphCLS(2, args.dim, 128, 2, encoder, False, dp=0.2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)

train_acc = []
test_acc = []
train_acc.append(evaluate(model, train_loader, device))
test_acc.append(evaluate(model, test_loader, device))
print(f'Epoch [0/{epochs}], acc_train: {train_acc[-1]:.6f}, acc_test: {test_acc[-1]:.6f}')
for epoch in range(epochs):
    train(model, train_loader, optimizer, lr_scheduler, device, epoch)
    train_acc.append(evaluate(model, train_loader, device))
    test_acc.append(evaluate(model, test_loader, device))
    print(f'Epoch [{epoch+1}/{epochs}], acc_train: {train_acc[-1]:.6f}, acc_test: {test_acc[-1]:.6f}')

