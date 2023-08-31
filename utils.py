import torch
import pandas as pd
import numpy as np
from datetime import datetime


class IdentityEncoder(object):
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)

class DateTimeEncoder(object):
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(pd.to_datetime(df).apply(lambda x: datetime.timestamp(x)).values).view(-1, 1).to(self.dtype)

class LocationEncoder(object):
    def __init__(self, dtype=None):
        self.dtype = dtype
    
    def __call__(self, df):
        x = torch.zeros(len(df), 2)
        for id, loc in enumerate(df.values):
            lat, long = eval(loc)
            x[id, 0] = lat
            x[id, 1] = long
        return x.to(self.dtype)

class SessionIdEncoder(object):
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.str.split('-',expand=True)[0].apply(hex_to_dec).values).view(-1, 1).to(self.dtype)

def hex_to_dec(id):
  if id == "" or id is None: return 0
  else: return int(id, 16)

def load_node_csv(dataframe, index_col, encoders=None):
    df = dataframe.set_index(index_col)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping

def load_edge_csv(dataframe, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None):
    df = dataframe

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])
    
    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr

def dense_adj(data):
  adj_dict = {}
  for node_i in data.num_nodes_dict.keys():
    adj_dict[node_i] = {}
    for node_j in data.num_nodes_dict.keys():
      adj_dict[node_i][node_j] = torch.from_numpy(np.zeros((data.num_nodes_dict[node_i], data.num_nodes_dict[node_j])))
  for key in data.edge_index_dict.keys():
    a,_,b = key
    for (i,j) in data.edge_index_dict[key].numpy().transpose():
      adj_dict[a][b][i][j] = 1
    return adj_dict