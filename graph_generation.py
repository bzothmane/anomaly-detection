import torch
import pandas as pd
import json
import pickle as pk
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
from datetime import datetime
import numpy as np
import os
from os import walk


def encode_column(df,col,encoder,checkpoint):
    if col in checkpoint.keys():
        init_batch= checkpoint[col]['batch']
        with open(str(col)+"_encoding.pkl", 'rb') as f:
            var =pk.load(f)
    else:
        init_batch=0
        var = encoder(df.iloc[0:100][col])
    for batch in range(init_batch+1,int(len(df)/100)):
        print(batch)
        var=torch.cat((var, encoder(df.iloc[batch*100:min((batch+1)*100,len(df))][col])), 1)                
        with open(str(col)+"_encoding.pkl", 'wb') as f:
            pk.dump(var, f)
        checkpoint[col]={'batch':batch}
        with open("checkpoint.pkl", 'wb') as f:
            pk.dump(checkpoint, f)
        print(checkpoint)
    return var



def list_files(path):
    f = []
    for (dirpath, dirnames, filenames) in walk(path):
        f.extend(filenames)
        break
    return f





def load_node_csv(dataframe, index_col,path, encoders=None):
    df = dataframe.set_index(index_col)
    mapping = {index: i for i, index in enumerate(df.index.unique())}
    x = None
    # if list_col is not None:
    #    for col in list_col:
    #        xs = torch.tensor(df[col].values)
    #        x = torch.cat([x, xs], dim=1)
    if encoders is not None:
        #xs = [encoder(df[col]) for col, encoder in encoders.items()]
        xs=[]

        if 'checkpoint.pkl' in list_files(path):
            print('get_the_checkpoint')
            with open('checkpoint.pkl', 'rb') as f:
                checkpoint =pk.load(f)
        else :
            checkpoint={}
    
        for col, encoder in encoders.items():
            print(col)
            var= encode_column(df,col,encoder,checkpoint)
            xs.append(var)
        x = torch.cat(xs, dim=-1)

    return x, mapping



class SequenceEncoder(object):
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
          
        return x.cpu()


class GenresEncoder(object):
    def __init__(self, sep='|'):
        self.sep = sep

    def __call__(self, df):
        genres = set(g for col in df.values for g in col.split(self.sep))
        mapping = {genre: i for i, genre in enumerate(genres)}

        x = torch.zeros(len(df), len(mapping))
        for i, col in enumerate(df.values):
            for genre in col.split(self.sep):
                x[i, mapping[genre]] = 1
        return x


class IdentityEncoder(object):
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)

class DateTimeEncoder(object):
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(pd.to_datetime(df, infer_datetime_format=True).apply(lambda x: datetime.timestamp(x)).values).view(-1, 1).to(self.dtype)


def load_edge_csv(dataframe, src_index_col, src_mapping, dst_index_col, dst_mapping,path,
                  encoders=None):
    df = dataframe

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        if 'checkpoint.pkl' in list_files(path):
            with open('checkpoint.pkl', 'rb') as f:
                checkpoint =pk.load(f)
        else :
            checkpoint={}
        #edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attrs=[]
        for col, encoder in encoders.items():
            print(col)
            var_edge= encode_column(df,col,encoder,checkpoint)
            edge_attrs.append(var_edge)
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr

def generate_graph(dataframe,customer_encodings,product_encodings,session_encodings, licence_encodings,path ):
    data = HeteroData()


    # Loading nodes into graph
    data['customer'].x, customer_mapping = load_node_csv(dataframe, "Customer_id",path,customer_encodings)
    data['product'].x, product_mapping = load_node_csv(dataframe, "product_id",path, product_encodings)
    _, user_mapping = load_node_csv(dataframe, "user_id",path)


    data['user'].num_nodes = len(user_mapping)  # user has no features

    print(data)

    # Loading edges into graph
    data['customer', 'has', 'user'].edge_index, _ = load_edge_csv(
        dataframe,
        src_index_col='Customer_id',
        src_mapping=customer_mapping,
        dst_index_col='user_id',
        dst_mapping=user_mapping,
        path=path
    )


    data['product', 'license','customer'].edge_index, data[
        'product', 'license','customer'].edge_attr = load_edge_csv(
        dataframe,
        src_index_col='product_id',
        src_mapping=product_mapping,
        dst_index_col='Customer_id',
        dst_mapping=customer_mapping,
        path=path,
        encoders=licence_encodings)



    data['user', 'opened_session', 'product'].edge_index, data[
        'user', 'opened_session', 'product'].edge_attr = load_edge_csv(
        dataframe,
        src_index_col='user_id',
        src_mapping=user_mapping,
        dst_index_col='product_id',
        dst_mapping=product_mapping,
        path=path,
        encoders=session_encodings
    )

    return data


def display_csv(path):
    df = pd.read_csv(path)
    print(df.head())


print("device")
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
print( "device", device)
database_path = "data_out_10000.csv"
fake="fake_data_2.csv"
print('get data')
dataframe = pd.read_csv(database_path).fillna("")
print(dataframe.shape)
fake = pd.read_csv(fake).fillna("")
dataframe=pd.concat([dataframe,fake])
path="/usr/users/gpusdi1/gpusdi1_39/Test"

customer_encodings = {"Location": GenresEncoder()}
product_encodings = {  # "brand": SequenceEncoder(),
        "price": IdentityEncoder(dtype=torch.long),
        "category_id": IdentityEncoder(dtype=torch.long)
}
session_encodings = {
        "Session_id": SequenceEncoder(),
        "Session_start_datetime": DateTimeEncoder(dtype=torch.long),
        "Session_end_datetime": DateTimeEncoder(dtype=torch.long),
        "event_type": SequenceEncoder(),
        "user_id": IdentityEncoder(dtype=torch.long)
}
licence_encodings = {
        "License_id": IdentityEncoder(dtype=torch.long),
        "License_start_date": DateTimeEncoder(dtype=torch.long),
        "License_end_date": DateTimeEncoder(dtype=torch.long),
}

print('generate graph')
data= generate_graph(dataframe,customer_encodings,product_encodings,session_encodings, licence_encodings,path)


print(data.edge_index_dict)

with open('graph.pkl', 'wb') as f:
    pk.dump(data, f)


with open('graph_index.pkl', 'wb') as f:
    pk.dump(data.edge_index_dict, f)





