import numpy as np
import torch
from torch.nn import Linear, Parameter
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import contains_self_loops

from torch_geometric.nn import HeteroConv, GCNConv, GATConv, Linear, to_hetero

#region G4SATBench
class G4GCN_LCG(nn.Module):
    def __init__(self, hidden_channels=8, num_conv_layers=4, include_meta_node=False):
        super().__init__()
        self.include_meta_node = include_meta_node
        if self.include_meta_node:
            self.lin0_m = Linear(1, hidden_channels, bias=False)
            self.lins_m = []
        self.lin0_c = Linear(1, hidden_channels, bias=False)
        self.lin0_v = Linear(1, hidden_channels, bias=False)
        self.convs = torch.nn.ModuleList()
        self.lins_c = []
        self.lins_v = []

        for _ in range(num_conv_layers):
            conv_dict = {
                ('clause', 'contains', 'literal'): G4GCNConv(*[hidden_channels]*3),
                # ('literal', 'paired_with', 'literal'): G4GCNConv(*[hidden_channels]*3), #Not used
                ('literal', 'rev_contains', 'clause'): G4GCNConv(*[hidden_channels]*3),
            }
            if self.include_meta_node:
                conv_dict[('meta', 'connects', 'clause')] = G4GCNConv(*[hidden_channels]*3)
                conv_dict[('clause', 'rev_connects', 'meta')] = G4GCNConv(*[hidden_channels]*3)
                self.lins_m.append(Linear(hidden_channels*2, hidden_channels))
                self.lins_c.append(Linear(hidden_channels*3, hidden_channels))
                self.lins_v.append(Linear(hidden_channels*3, hidden_channels))
            else:
                self.lins_c.append(Linear(hidden_channels*2, hidden_channels))
                self.lins_v.append(Linear(hidden_channels*3, hidden_channels))

            conv = HeteroConv(conv_dict, aggr='cat')
            self.convs.append(conv)
        
        self.lin_out= Linear(hidden_channels, 1)
    
    def forward(self, x_dict, deg_dict, edge_index_dict):
        if self.include_meta_node:
            x_dict['meta'] = self.lin0_m(x_dict['meta'])
        x_dict['clause'] = self.lin0_c(x_dict['clause'])
        x_dict['literal'] = self.lin0_v(x_dict['literal'])
        literala, literalb = edge_index_dict['literal', 'paired_with', 'literal']
        literal_pair_lookup = {literala.item(): literalb.item() for literala, literalb in zip(literala, literalb)}
        literal_pair_map = np.array([literal_pair_lookup[literal_idx] for literal_idx in range(deg_dict['literal'].shape[0])])
        for conv_layer, conv in enumerate(self.convs):
            x_dict_prev = x_dict
            x_dict = conv(x_dict, deg_dict, edge_index_dict)
            if self.include_meta_node:
                x_dict['meta'] = torch.cat([x_dict['meta'], x_dict_prev['meta']], dim=1)
                x_dict['meta'] = self.lins_m[conv_layer](x_dict['meta'])
            x_dict['clause'] = torch.cat([x_dict['clause'], x_dict_prev['clause']], dim=1)
            x_dict['literal'] = torch.cat([x_dict['literal'], x_dict_prev['literal'], x_dict_prev['literal'][literal_pair_map]], dim=1)
            x_dict['clause'] = self.lins_c[conv_layer](x_dict['clause'])
            x_dict['literal'] = self.lins_v[conv_layer](x_dict['literal'])
        out = torch.cat([x_dict['literal'],x_dict['literal'][literal_pair_map]], dim =1)
        return out

class G4GCN_VCG(nn.Module):
    def __init__(self, hidden_channels=8, num_conv_layers=4, include_meta_node=False):
        super().__init__()
        self.include_meta_node = include_meta_node
        if self.include_meta_node:
            self.lin0_m = Linear(1, hidden_channels, bias=False)
            self.lins_m = []
        self.lin0_c = Linear(1, hidden_channels, bias=False)
        self.lin0_v = Linear(1, hidden_channels, bias=False)
        self.convs = torch.nn.ModuleList()
        self.lins_c = []
        self.lins_v = []

        for _ in range(num_conv_layers):
            conv_dict = {
                ('clause', 'contains_pos', 'variable'): G4GCNConv(*[hidden_channels]*3),
                ('clause', 'contains_neg', 'variable'): G4GCNConv(*[hidden_channels]*3),
                ('variable', 'rev_contains_pos', 'clause'): G4GCNConv(*[hidden_channels]*3),
                ('variable', 'rev_contains_neg', 'clause'): G4GCNConv(*[hidden_channels]*3)
            }
            if self.include_meta_node:
                conv_dict[('meta', 'connects', 'clause')] = G4GCNConv(*[hidden_channels]*3)
                conv_dict[('clause', 'rev_connects', 'meta')] = G4GCNConv(*[hidden_channels]*3)
                self.lins_m.append(Linear(hidden_channels*2, hidden_channels))
                self.lins_c.append(Linear(hidden_channels*4, hidden_channels))
                self.lins_v.append(Linear(hidden_channels*3, hidden_channels))
            else:
                self.lins_c.append(Linear(hidden_channels*3, hidden_channels))
                self.lins_v.append(Linear(hidden_channels*3, hidden_channels))

            conv = HeteroConv(conv_dict, aggr='cat')
            self.convs.append(conv)
        
        self.lin_out= Linear(hidden_channels, 1)

    def forward(self, x_dict, deg_dict, edge_index_dict):
        if self.include_meta_node:
            x_dict['meta'] = self.lin0_m(x_dict['meta'])
        x_dict['clause'] = self.lin0_c(x_dict['clause'])
        x_dict['variable'] = self.lin0_v(x_dict['variable'])
        for conv_layer, conv in enumerate(self.convs):
            x_dict_prev = x_dict
            x_dict = conv(x_dict, deg_dict, edge_index_dict)
            if self.include_meta_node:
                x_dict['meta'] = torch.cat([x_dict['meta'], x_dict_prev['meta']], dim=1)
                x_dict['meta'] = self.lins_m[conv_layer](x_dict['meta'])
            x_dict['clause'] = torch.cat([x_dict['clause'], x_dict_prev['clause']], dim=1)
            x_dict['variable'] = torch.cat([x_dict['variable'], x_dict_prev['variable']], dim=1)
            x_dict['clause'] = self.lins_c[conv_layer](x_dict['clause'])
            x_dict['variable'] = self.lins_v[conv_layer](x_dict['variable'])
        out = x_dict['variable']
        return out

class G4GCNConv(MessagePassing):
    # Must manually ensure that models have the same parameters when initialized
    def __init__(self, src_channels, tgt_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin_src = Linear(src_channels, out_channels, bias=True)
        self.lin_src.reset_parameters()
        self.MLP = MLP([out_channels, out_channels, out_channels, out_channels])

    def forward(self, x, deg, edge_index):
        # Linearly transform node feature matrix.
        x_src, x_trg = x
        x_src = self.lin_src(x_src).relu()

        # Compute normalization.
        src, trg = edge_index
        deg_src, deg_trg = deg
        deg_src_inv_sqrt = deg_src.pow(-0.5)
        deg_src_inv_sqrt[deg_src_inv_sqrt == float('inf')] = 0
        deg_trg_inv_sqrt = deg_trg.pow(-0.5)
        deg_trg_inv_sqrt[deg_trg_inv_sqrt == float('inf')] = 0
        norm = deg_src_inv_sqrt[src] * deg_trg_inv_sqrt[trg]

        # Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)
        return out

    def message(self, x_i, x_j, norm):
        # x_j has shape [E, out_channels]

        # Normalize node features.
        return norm.view(-1, 1) * self.MLP(x_j).relu()


#endregion
class MLP(nn.Module):
    def __init__(self, layer_sizes):
        """
        layer_sizes is a list of integers with the input size as the 
        first element and the output size as the last element
        """
        super().__init__()
        self.layer_sizes = layer_sizes
        layers = []
        for layer, layer_size in enumerate(layer_sizes[:-1]):
            layers.append(Linear(layer_sizes[layer], layer_sizes[layer+1], bias=True))
            if layer != len(layer_sizes[:-1]) - 1:
                layers.append(nn.ReLU())
        self.linear_layers = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.linear_layers(x)
        return x

#region Not Used
class custom_GCNConv_same_as_builtin(MessagePassing):
    # Must manually ensure that models have the same parameters when initialized
    def __init__(self, in_channels, out_channels, normalize = True, add_self_loops=True):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index, edge_weight):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix. 
        if self.add_self_loops and not contains_self_loops(edge_index):
            edge_index, edge_weight = add_self_loops(edge_index, edge_attr=edge_weight, fill_value=1.)

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        if self.normalize:
            num_nodes = edge_index.max().item() + 1
            row, col = edge_index
            # deg = degree(col, x.size(0), dtype=x.dtype)
            deg = torch.zeros(num_nodes, dtype=edge_weight.dtype)
            deg.scatter_add_(0, col, edge_weight)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        else:
            norm = torch.ones(edge_index.shape[1], dtype=edge_weight.dtype)

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm, edge_weight=edge_weight)

        # Step 6: Apply a final bias vector.
        out = out + self.bias

        return out

    def message(self, x_j, norm, edge_weight):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j * edge_weight.view(-1, 1)
#endregion
