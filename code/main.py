from torch_geometric.datasets import *
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from models import *

# 超参数
# datasets: [cora, citeseer, pubmed, nell]
datasets_name = 'pubmed'
root='datasets'
seed = 41

np.random.seed(seed)
torch.manual_seed(seed)

if datasets_name in ['cora', 'citeseer', 'pubmed']:
    dataset = Planetoid(root, name=datasets_name)
elif datasets_name in ['nell']:
    dataset = NELL(root)
elif datasets_name in ['MUTAG']:
    dataset = Entities(root, name=datasets_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ChebyNet(dataset).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in tqdm(range(200)):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print("Datasets: {}, Models: {}".format(datasets_name, model._get_name()))
print(f'Accuracy: {acc:.4f}')
