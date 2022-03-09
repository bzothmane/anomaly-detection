
import torch
import pandas as pd
import json
import pickle as pk


from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
from torch_geometric.nn import MetaPath2Vec

with open('graph_index.pkl', 'rb') as f:
    graph_index =pk.load(f)

print(graph_index)

metapath = [
    ('customer', 'has', 'user'),
    ('user', 'opened_session', 'product'),
    ('product', 'license','customer')
  
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MetaPath2Vec(graph_index, embedding_dim=128,
                     metapath=metapath, walk_length=50, context_size=7,
                     walks_per_node=2, num_negative_samples=5,
                     sparse=True).to(device)

loader = model.loader(batch_size=128, shuffle=True, num_workers=6)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)


def train(epoch, log_steps=100, eval_steps=2000):
    model.train()

    total_loss = 0
    for i, (pos_rw, neg_rw) in enumerate(loader):
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % log_steps == 0:
            print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                   f'Loss: {total_loss / log_steps:.4f}'))
            total_loss = 0


for epoch in range(1, 6):
  train(epoch)

embedding=model('customer')
print(type(embedding))


with open('embedding.pkl', 'wb') as f:
    pk.dump(embedding, f)


