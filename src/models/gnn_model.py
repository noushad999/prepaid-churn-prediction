import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
from torch_geometric.data import Data, Batch

class GNNChurnPredictor(nn.Module):
    """
    Graph Neural Network for churn prediction using call network topology.
    Supports GCN, GAT, and GraphSAGE architectures.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.input_dim = config['input_dim']
        self.hidden_channels = config['hidden_channels']
        self.num_layers = config['num_layers']
        self.conv_type = config.get('conv_type', 'GCN')
        self.heads = config.get('heads', 4)
        self.dropout = config.get('dropout', 0.3)
        self.num_classes = config.get('num_classes', 2)
        
        # Build convolutional layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        if self.conv_type == 'GCN':
            self.convs.append(GCNConv(self.input_dim, self.hidden_channels))
        elif self.conv_type == 'GAT':
            self.convs.append(GATConv(
                self.input_dim, self.hidden_channels,
                heads=self.heads, dropout=self.dropout
            ))
        elif self.conv_type == 'GraphSAGE':
            self.convs.append(SAGEConv(self.input_dim, self.hidden_channels))
        
        self.batch_norms.append(nn.BatchNorm1d(self.hidden_channels * (self.heads if self.conv_type == 'GAT' else 1)))
        
        # Hidden layers
        hidden_dim = self.hidden_channels * (self.heads if self.conv_type == 'GAT' else 1)
        for _ in range(self.num_layers - 2):
            if self.conv_type == 'GCN':
                self.convs.append(GCNConv(hidden_dim, self.hidden_channels))
            elif self.conv_type == 'GAT':
                self.convs.append(GATConv(
                    hidden_dim, self.hidden_channels,
                    heads=self.heads, dropout=self.dropout
                ))
            elif self.conv_type == 'GraphSAGE':
                self.convs.append(SAGEConv(hidden_dim, self.hidden_channels))
            
            self.batch_norms.append(nn.BatchNorm1d(self.hidden_channels * (self.heads if self.conv_type == 'GAT' else 1)))
        
        # Output layer
        if self.conv_type == 'GCN':
            self.convs.append(GCNConv(hidden_dim, self.hidden_channels))
        elif self.conv_type == 'GAT':
            self.convs.append(GATConv(
                hidden_dim, self.hidden_channels,
                heads=1, concat=False, dropout=self.dropout
            ))
        elif self.conv_type == 'GraphSAGE':
            self.convs.append(SAGEConv(hidden_dim, self.hidden_channels))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_channels, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.num_classes)
        )
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else None
        
        # Graph convolutions
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final convolution
        x = self.convs[-1](x, edge_index)
        
        # Global pooling (if batch is provided)
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        # Classification
        out = self.classifier(x)
        
        return out
