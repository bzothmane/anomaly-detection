import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero

def loss_func_train(attrs, X_hat, adj, A_hat, alpha=0.8):
    # Attribute reconstruction loss
    diff_attribute = []
    for key in attrs.keys():
        diff_attribute.append(torch.pow(X_hat[key] - attrs[key], 2))
    diff_attribute = torch.cat(tuple(diff_attribute), 0)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    # structure reconstruction loss
    diff_structure_all = []
    for key1 in adj.keys():
        structure = []
        for key2 in adj.keys():
            structure.append(torch.pow(A_hat[key1][key2] - adj[key1][key2], 2))
        diff_structure_all.append(torch.cat(tuple(structure), 1))
    diff_structure = torch.cat(tuple(diff_structure_all), 0)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    structure_cost = torch.mean(structure_reconstruction_errors)

    cost =  alpha * attribute_reconstruction_errors + (1-alpha) * structure_reconstruction_errors

    return cost, structure_cost, attribute_cost

def loss_func_test(attrs, X_hat):
    # Attribute reconstruction loss
    diff_attribute = torch.pow(X_hat - attrs, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    # structure reconstruction loss
    # diff_structure = torch.pow(A_hat - adj, 2)
    # structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    # structure_cost = torch.mean(structure_reconstruction_errors)
    structure_cost = 0

    cost =  attribute_reconstruction_errors


    return cost, structure_cost, attribute_cost

class Encoder(nn.Module):
    def __init__(self, num_features, hidden_channels, dropout):
        super().__init__()
        self.conv1 = SAGEConv((num_features, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index).relu()
        return x

class Attribute_Decoder(nn.Module):
    def __init__(self, num_features, hidden_channels, dropout):
        super().__init__()

        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), num_features)
        self.dropout = dropout

    def forward(self, x, adj):

        x = F.relu(self.conv1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv2(x, adj))

        return x

class Structure_Decoder(nn.Module):
    def __init__(self, num_nodes, hidden_channels, dropout):
        super().__init__()

        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), num_nodes)
        self.dropout = dropout

    def forward(self, x, adj):

        x = F.relu(self.conv1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv2(x, adj))

        return x.T


class Dominant(nn.Module):
    def __init__(self, feat_size, hidden_size, num_nodes_dict, dropout, metadata):
        super().__init__()
        
        self.shared_encoder = to_hetero(Encoder(feat_size, hidden_size, dropout), metadata, aggr='sum')
        self.attr_decoder = to_hetero(Attribute_Decoder(feat_size, hidden_size, dropout), metadata, aggr='sum')
        self.struct_decoder_dict = {}
        for key in num_nodes_dict.keys():
          self.struct_decoder_dict[key] = to_hetero(Structure_Decoder(num_nodes_dict[key], hidden_size, dropout), metadata, aggr='sum')
    
    def forward(self, x_dict, adj_dict):

        # encode
        x_dict = self.shared_encoder(x_dict, adj_dict)
        # decode feature matrix
        x_hat_dict = self.attr_decoder(x_dict, adj_dict)
        # decode adjacency matrix
        struct_reconstructed_dict={}
        for key in self.struct_decoder_dict.keys():
          struct_reconstructed_dict[key] = self.struct_decoder_dict[key](x_dict, adj_dict)
        # return reconstructed matrices
        return struct_reconstructed_dict, x_hat_dict