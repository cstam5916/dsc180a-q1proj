import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SMPNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.num_layers = num_layers

        # Initial Map to Latenet Space
        self.linear_start = nn.Linear(in_channels, hidden_channels)

        # GCN Layers and Norms
        self.layernorms_gcn = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])
        self.convs = nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers)])
        self.alphas_gcn = nn.ParameterList([nn.Parameter(torch.tensor(1e-6)) for _ in range(num_layers)])

        # Feedforward Layers
        self.layernorms_ff = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])
        self.ffw = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels) for _ in range(num_layers)])
        self.alphas_ff = nn.ParameterList([nn.Parameter(torch.tensor(1e-6)) for _ in range(num_layers)])

        self.linear_end = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.silu(self.linear_start(x))
        for i in range(self.num_layers):
            x = self.forward_gcn(x, edge_index, i)
            # x = self.pointwise_ff(x, i)
        x = F.log_softmax(self.linear_end(x), dim=1)
        return x

    def forward_gcn(self, x, edge_index, layer_idx):
        conv_x = self.layernorms_gcn[layer_idx](x)
        conv_x = self.convs[layer_idx](conv_x, edge_index)
        conv_x = F.silu(conv_x)
        x = (self.alphas_gcn[layer_idx] * conv_x) + x
        return x

    def pointwise_ff(self, x, layer_idx):
        norm_x = self.layernorms_ff[layer_idx](x)
        x = self.alphas_ff[layer_idx] * F.silu(self.ffw[layer_idx](norm_x)) + x
        return x

# import torch
# from torch import nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv

# class SMPNN(torch.nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.layernorm_1 = nn.LayerNorm(in_channels)
#         self.conv = GCNConv(in_channels, in_channels)
#         self.alpha_1 = nn.Parameter(data=torch.tensor(1e-6))

#         self.layernorm_2 = nn.LayerNorm(in_channels)
#         self.ffw = nn.Parameter(torch.randn(in_channels))
#         self.alpha_2 = nn.Parameter(data=torch.tensor(1e-6))

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = self.forward_gcn(x, edge_index)
#         x = self.pointwise_ff(x)
#         return x

#     def forward_gcn(self, x, edge_index):
#         conv_x = self.layernorm_1(x)
#         conv_x = F.silu(self.conv(conv_x, edge_index))
#         x += self.alpha_1 * conv_x
#         return F.log_softmax(x, dim=1)
    
#     def pointwise_ff(self, x):
#         norm_x = self.layernorm_2(x)
#         x = self.alpha_2 * F.silu(norm_x * self.ffw) + x
#         return x