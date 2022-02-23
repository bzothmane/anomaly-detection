import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData


def load_node_csv(dataframe, index_col, encoders=None):
    df = dataframe.set_index(index_col)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    # if list_col is not None:
    #    for col in list_col:
    #        xs = torch.tensor(df[col].values)
    #        x = torch.cat([x, xs], dim=1)
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
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


def generate_graph(dataframe):
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
    data['customer'].x, customer_mapping = load_node_csv(dataframe, "Customer_id", customer_encodings)
    data['product'].x, product_mapping = load_node_csv(dataframe, "product_id", product_encodings)
    _, user_mapping = load_node_csv(dataframe, "user_id")
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

    data['customer', 'owns_license', 'product'].edge_index, data[
        'customer', 'owns_license', 'product'].edge_attr = load_edge_csv(
        dataframe,
        src_index_col='Customer_id',
        src_mapping=customer_mapping,
        dst_index_col='product_id',
        dst_mapping=product_mapping,
        encoders=licence_encodings
    )
    data['user', 'opened_session', 'product'].edge_index, data[
        'user', 'opened_session', 'product'].edge_attr = load_edge_csv(
        dataframe,
        src_index_col='user_id',
        src_mapping=user_mapping,
        dst_index_col='product_id',
        dst_mapping=product_mapping,
        encoders=session_encodings
    )
    return data


def display_csv(path):
    df = pd.read_csv(path)
    print(df.head())


if __name__ == '__main__':
    database_path = "data_out_head_head.csv"
    dataframe = pd.read_csv(database_path).fillna("")

    customer_encodings = {"Location": GenresEncoder()}
    product_encodings = {  # "brand": SequenceEncoder(),
        "price": IdentityEncoder(dtype=torch.long),
        "category_id": IdentityEncoder(dtype=torch.long),
        # "category_code": SequenceEncoder()
    }
    session_encodings = {
        "Session_id": SequenceEncoder(),
        # "Session_start_datetime": IdentityEncoder(dtype=torch.long),
        # "Session_end_datetime": IdentityEncoder(dtype=torch.long),
        "event_type": SequenceEncoder(),
        # "user_id": IdentityEncoder(dtype=torch.long)
    }
    licence_encodings = {
        "License_id": IdentityEncoder(dtype=torch.long),
        # "License_start_date": IdentityEncoder(dtype=torch.long),
        # "License_end_date": IdentityEncoder(dtype=torch.long),
    }

    customer_graphs = {}
    for c in list(dataframe["Customer_id"].unique()):
        df = dataframe[dataframe["Customer_id"] == c]
        customer_graphs[c] = generate_graph(df)
    
    print(customer_graphs)
#
#
# ['event_time',
#  'event_type',
#  'product_id',
#  'category_id',
#  'category_code',
#  'brand',
#  'price',
#  'user_id',
#  'Session_id',
#  'Customer_id',
#  'Location',
#  'License_id',
#  'Session_start_datetime',
#  'Session_end_datetime',
#  'duration',
#  'License_start_date',
#  'License_end_date']
