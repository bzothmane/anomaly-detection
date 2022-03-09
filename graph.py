import torch
import pandas as pd
import json
import pickle as pk
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
from datetime import datetime
import numpy as np
import os

def load_node_csv(dataframe, index_col, encoders=None):
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
        for col, encoder in encoders.items():
            print(col)
            var=encoder(df.iloc[0:100][col])
            for batch in range(1,int(len(df)/100)):
                print(batch)
                var=torch.cat((var, encoder(df.iloc[batch*100:min((batch+1)*100,len(df))][col])), 1) 
            with open(str(col)+"_encoding.pkl", 'wb') as f:
                pk.dump(var, f)
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


def load_edge_csv(dataframe, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None):
    df = dataframe

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        #edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attrs=[]
        for col, encoder in encoders.items():
            print(col)
            os.mkdir(str(col))
            var_edge=encoder(df.iloc[0:100][col])
            with open(str(col)+"/"+str(col)+"_encoding_0.pkl", 'wb') as f:
                pk.dump(var_edge, f)
            for batch in range(1,int(len(df)/100)):
                print(batch)
                add=encoder(df.iloc[batch*100:min((batch+1)*100,len(df))][col])
                with open(str(col)+"/"+str(col)+"_encoding_"+str(batch)+".pkl", 'wb') as f:
                    pk.dump(add, f)
                var_edge=torch.cat((var_edge,add),1)        
            with open(str(col)+"_encoding.pkl", 'wb') as f:
                pk.dump(var_edge, f)
            edge_attrs.append(var_edge)
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr

def generate_graph(dataframe,customer_encodings,product_encodings,session_encodings, licence_encodings ):
    data = HeteroData()

    # if df_full:
    #     customer_encodings = {"Location": GenresEncoder()}
    #     product_encodings = {  # "brand": SequenceEncoder(),
    #         "price": IdentityEncoder(dtype=torch.long),
    #         "category_id": IdentityEncoder(dtype=torch.long),
    #         # "category_code": SequenceEncoder()
    #     }
    #     session_encodings = {
    #         "Session_id": SequenceEncoder(),
    #         # "Session_start_datetime": IdentityEncoder(dtype=torch.long),
    #         # "Session_end_datetime": IdentityEncoder(dtype=torch.long),
    #         "event_type": SequenceEncoder(),
    #         # "user_id": IdentityEncoder(dtype=torch.long)
    #     }
    #     licence_encodings = {
    #         "License_id": IdentityEncoder(dtype=torch.long),
    #         # "License_start_date": IdentityEncoder(dtype=torch.long),
    #         # "License_end_date": IdentityEncoder(dtype=torch.long),
    #     }
    # else:
    #     customer_encodings = None
    #     product_encodings = {"category_id": IdentityEncoder(dtype=torch.long),
    #                          "price": IdentityEncoder(dtype=torch.long)}

    # Loading nodes into graph
    data['customer'].x, customer_mapping = load_node_csv(dataframe, "Customer_id",customer_encodings)
    with open('data_customer_encoding.pkl', 'wb') as f:
        pk.dump(data['customer'].x, f)
    with open('data_customer_mapping.pkl', 'wb') as f:
        pk.dump(customer_mapping, f)
    data['product'].x, product_mapping = load_node_csv(dataframe, "product_id", product_encodings)
    with open('data_product_encoding.pkl', 'wb') as f:
        pk.dump(data['product'].x, f)
    with open('data_product_mapping.pkl', 'wb') as f:
        pk.dump(product_mapping, f)
    _, user_mapping = load_node_csv(dataframe, "user_id")
 
    with open('data_user_encoding.pkl', 'wb') as f:
        pk.dump(_, f)
    with open('data_user_mapping.pkl', 'wb') as f:
        pk.dump(user_mapping, f)

    data['user'].num_nodes = len(user_mapping)  # user has no features

    print(data)

    # Loading edges into graph
    data['customer', 'has', 'user'].edge_index, _ = load_edge_csv(
        dataframe,
        src_index_col='Customer_id',
        src_mapping=customer_mapping,
        dst_index_col='user_id',
        dst_mapping=user_mapping,
    )
    with open('customer_user_edges.pkl', 'wb') as f:
        pk.dump(data['customer', 'has', 'user'].edge_index, f)

    data['product', 'license','customer'].edge_index, data[
        'product', 'license','customer'].edge_attr = load_edge_csv(
        dataframe,
        src_index_col='product_id',
        src_mapping=product_mapping,
        dst_index_col='Customer_id',
        dst_mapping=customer_mapping,
        encoders=licence_encodings)
    with open('product_license_customer_edges_index.pkl', 'wb') as f:
        pk.dump(data['product', 'license','customer'].edge_index, f)
    with open('product_license_customer_edges_attr.pkl', 'wb') as f:
        pk.dump(data['product', 'license','customer'].edge_attr, f)


    data['user', 'opened_session', 'product'].edge_index, data[
        'user', 'opened_session', 'product'].edge_attr = load_edge_csv(
        dataframe,
        src_index_col='user_id',
        src_mapping=user_mapping,
        dst_index_col='product_id',
        dst_mapping=product_mapping,
        encoders=session_encodings
    )
    with open('user_session_product_edges_index.pkl', 'wb') as f:
        pk.dump(data['user', 'opened_session', 'product'].edge_index, f)
    with open('user_session_product_edges_attr.pkl', 'wb') as f:
        pk.dump(data['user', 'opened_session', 'product'].edge_attr, f)
    return data


def display_csv(path):
    df = pd.read_csv(path)
    print(df.head())


print("device")
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
print( "device", device)
database_path = "data_out_1000.csv"
print('get data')
dataframe = pd.read_csv(database_path).fillna("")
print(dataframe.shape)

customer_encodings = {"Location": GenresEncoder()}
product_encodings = {  # "brand": SequenceEncoder(),
        "price": IdentityEncoder(dtype=torch.long),
        "category_id": IdentityEncoder(dtype=torch.long),
        # "category_code": SequenceEncoder()
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
data= generate_graph(dataframe,customer_encodings,product_encodings,session_encodings, licence_encodings)


print(data.edge_index_dict)

with open('graph.pkl', 'wb') as f:
    pk.dump(data, f)


with open('graph_index.pkl', 'wb') as f:
    pk.dump(data.edge_index_dict, f)





