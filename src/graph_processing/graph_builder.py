import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)

class CallNetworkGraphBuilder:
    """
    Build call network graphs for GNN-based churn prediction.
    Analyzes customer relationships and network influence.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.graph = None
        self.node_features = None
        self.edge_index = None
        
    def build_from_call_data(self, call_records: pd.DataFrame,
                            customer_features: pd.DataFrame) -> Data:
        """
        Build graph from call detail records (CDR).
        
        Args:
            call_records: DataFrame with columns ['caller_id', 'receiver_id', 'duration', 'timestamp']
            customer_features: DataFrame with customer feature vectors
        """
        logger.info("Building call network graph...")
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add edges from call records
        for _, row in call_records.iterrows():
            caller = row['caller_id']
            receiver = row['receiver_id']
            duration = row.get('duration', 1)
            
            if G.has_edge(caller, receiver):
                G[caller][receiver]['weight'] += duration
                G[caller][receiver]['call_count'] += 1
            else:
                G.add_edge(caller, receiver, weight=duration, call_count=1)
        
        # Add node features
        for node in G.nodes():
            if node in customer_features.index:
                G.nodes[node]['features'] = customer_features.loc[node].values
            else:
                # Default features for missing nodes
                G.nodes[node]['features'] = np.zeros(customer_features.shape[1])
        
        self.graph = G
        logger.info(f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Convert to PyTorch Geometric format
        return self._to_pytorch_geometric(G, customer_features)
    
    def _to_pytorch_geometric(self, G: nx.Graph,
                             customer_features: pd.DataFrame) -> Data:
        """Convert NetworkX graph to PyTorch Geometric Data object."""
        
        # Create node mapping
        node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
        reverse_mapping = {idx: node for node, idx in node_mapping.items()}
        
        # Extract node features
        node_features = []
        for idx in range(len(node_mapping)):
            node_id = reverse_mapping[idx]
            features = G.nodes[node_id]['features']
            node_features.append(features)
        
        node_features = torch.tensor(np.array(node_features), dtype=torch.float)
        
        # Extract edges
        edge_list = []
        edge_weights = []
        for u, v, data in G.edges(data=True):
            u_idx = node_mapping[u]
            v_idx = node_mapping[v]
            edge_list.append([u_idx, v_idx])
            edge_list.append([v_idx, u_idx])  # Undirected graph
            weight = data.get('weight', 1.0)
            edge_weights.extend([weight, weight])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        
        # Create PyG Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_weights.unsqueeze(1),
            num_nodes=len(node_mapping)
        )
        
        self.edge_index = edge_index
        self.node_features = node_features
        
        return data
    
    def extract_graph_features(self, customer_ids: List) -> pd.DataFrame:
        """
        Extract graph-based features for each customer.
        Features include: degree centrality, betweenness, PageRank, clustering coefficient.
        """
        logger.info("Extracting graph features...")
        
        if self.graph is None:
            raise ValueError("Graph not built. Call build_from_call_data first.")
        
        features = {}
        
        # Degree centrality
        degree_cent = nx.degree_centrality(self.graph)
        
        # Betweenness centrality
        betweenness_cent = nx.betweenness_centrality(self.graph)
        
        # PageRank
        pagerank = nx.pagerank(self.graph)
        
        # Clustering coefficient
        clustering = nx.clustering(self.graph)
        
        # Community detection (Louvain)
        try:
            from networkx.algorithms import community
            communities = community.greedy_modularity_communities(self.graph)
            community_map = {}
            for idx, comm in enumerate(communities):
                for node in comm:
                    community_map[node] = idx
        except:
            community_map = {node: 0 for node in self.graph.nodes()}
        
        # Compile features
        for customer_id in customer_ids:
            if customer_id in self.graph:
                features[customer_id] = {
                    'degree_centrality': degree_cent.get(customer_id, 0),
                    'betweenness_centrality': betweenness_cent.get(customer_id, 0),
                    'pagerank': pagerank.get(customer_id, 0),
                    'clustering_coefficient': clustering.get(customer_id, 0),
                    'community_id': community_map.get(customer_id, -1),
                    'degree': self.graph.degree(customer_id),
                }
            else:
                # Default values for customers not in graph
                features[customer_id] = {
                    'degree_centrality': 0,
                    'betweenness_centrality': 0,
                    'pagerank': 0,
                    'clustering_coefficient': 0,
                    'community_id': -1,
                    'degree': 0,
                }
        
        return pd.DataFrame.from_dict(features, orient='index')
