import torch
import torch.nn as nn
from typing import Dict, Optional
import numpy as np

class EnsembleChurnPredictor(nn.Module):
    """
    Ensemble model combining CCP-Net, GNN, and NLP models.
    Industry-standard multi-modal fusion for maximum accuracy.
    """
    
    def __init__(self, ccp_net, gnn_model, nlp_model, config: Dict):
        super().__init__()
        
        self.ccp_net = ccp_net
        self.gnn_model = gnn_model
        self.nlp_model = nlp_model
        
        self.use_gnn = config.get('use_gnn', True)
        self.use_nlp = config.get('use_nlp', True)
        self.dropout = config.get('dropout', 0.3)
        
        # Calculate fusion input dimension
        fusion_dim = 2  # CCP-Net output dim (num_classes)
        if self.use_gnn:
            fusion_dim += 2  # GNN output dim
        if self.use_nlp:
            fusion_dim += 128  # NLP feature dim
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(self.dropout),
            nn.Linear(128, 2)
        )
        
    def forward(self, tabular_data, graph_data=None, text_data=None):
        # CCP-Net prediction
        ccp_output = self.ccp_net(tabular_data)
        
        outputs = [ccp_output]
        
        # GNN prediction (if available)
        if self.use_gnn and graph_data is not None:
            gnn_output = self.gnn_model(graph_data)
            outputs.append(gnn_output)
        
        # NLP features (if available)
        if self.use_nlp and text_data is not None:
            _, nlp_features = self.nlp_model(
                text_data['input_ids'],
                text_data['attention_mask']
            )
            outputs.append(nlp_features)
        
        # Concatenate all outputs
        fused = torch.cat(outputs, dim=1)
        
        # Final prediction
        final_output = self.fusion(fused)
        
        return final_output
