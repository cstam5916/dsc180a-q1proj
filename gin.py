import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        mlp = torch.nn.Sequential(
            Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            Linear(hidden_channels, hidden_channels),
        )
        mlp_2 = torch.nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            Linear(hidden_channels, out_channels),
        )
        self.conv1 = GINConv(mlp, train_eps=True)
        self.conv2 = GINConv(mlp_2, train_eps=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    

class GINGraphLev(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        super().__init__()
        mlp = torch.nn.Sequential(
            Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            Linear(hidden_channels, hidden_channels),
        )
        mlp_2 = torch.nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            Linear(hidden_channels, hidden_channels),
        )
        self.conv1 = GINConv(mlp, train_eps=True)
        self.conv2 = GINConv(mlp_2, train_eps=True)
        self.linear = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return F.log_softmax(x, dim=-1)