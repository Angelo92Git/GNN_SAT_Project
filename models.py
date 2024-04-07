import torch
from torch.nn import Linear, Parameter
import torch.nn.functional as funcs
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import contains_self_loops
from torch_geometric.utils import to_networkx

from torch_geometric.nn import HeteroConv, GCNConv, GATConv, Linear, to_hetero

#region G4SATBench

class G4GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin0_c = Linear(1, 2)
        self.lin0_v = Linear(1, 2)
        self.lins_c = []
        self.lins_v = []
        self.lin_out = Linear(2, 1)
        self.convs = torch.nn.ModuleList()
        for _ in range(4):
            conv = HeteroConv({
                ('clause', 'contains_pos', 'variable'): G4GCNConv(),
                ('clause', 'contains_neg', 'variable'): G4GCNConv(),
                ('variable', 'rev_contains_pos', 'clause'): G4GCNConv(),
                ('variable', 'rev_contains_neg', 'clause'): G4GCNConv()
            }, aggr='cat')
            self.convs.append(conv)
            self.lins_c.append(Linear(6, 2))
            self.lins_v.append(Linear(6, 2))

    def forward(self, x_dict, deg_dict, edge_index_dict):
        x_dict['clause'] = self.lin0_c(x_dict['clause'])
        x_dict['variable'] = self.lin0_v(x_dict['variable'])
        for layer, conv in enumerate(self.convs):
            x_dict_prev = x_dict
            x_dict = conv(x_dict, deg_dict, edge_index_dict)
            x_dict['clause'] = torch.cat([x_dict['clause'], x_dict_prev['clause']], dim=1)
            x_dict['variable'] = torch.cat([x_dict['variable'], x_dict_prev['variable']], dim=1)
            x_dict['clause'] = self.lins_c[layer](x_dict['clause'])
            x_dict['variable'] = self.lins_v[layer](x_dict['variable'])
        out = x_dict['variable']
        return out

class G4GCNConv(MessagePassing):
    # Must manually ensure that models have the same parameters when initialized
    def __init__(self, src_channels=2, tgt_channels=2, out_channels=2):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin_src = Linear(src_channels, out_channels, bias=True)
        self.lin_src.reset_parameters()

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
        return norm.view(-1, 1) * x_j + x_i

class G4GCN_Lin(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = Linear(1, 2, bias=False)

    def forward(self, x):
        x = self.lin(x)
        return x

#endregion
class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lin1 = Linear(input_size, hidden_size)
        self.lin2 = Linear(hidden_size, hidden_size)
        self.lin3 = Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        x = x.relu()
        x = self.lin3(x)
        x = x.sigmoid()
        return x

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin0 = Linear(-1, hidden_channels, bias=False)
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels, bias=False)
        self.conv2 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin2 = Linear(-1, hidden_channels, bias=False)
        self.conv3 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin3 = Linear(-1, hidden_channels, bias=False)
        self.conv4 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin4 = Linear(-1, hidden_channels, bias=False)

    def forward(self, x, edge_index, edge_attr):
        x = self.lin0(x)
        x = self.conv1(x, edge_index, edge_attr) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr) + self.lin2(x)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr) + self.lin3(x)
        x = x.relu()
        x = self.conv4(x, edge_index, edge_attr) + self.lin4(x)
        x = x.sigmoid()
        return x



#region Not Useful
class custom_GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x).relu()

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm, edge_attr=edge_attr)

        # Step 6: Apply a final bias vector.
        out = out + self.bias

        return out

    def message(self, x_j, norm, edge_attr):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j * edge_attr.view(-1, 1)

    # def update(self, aggr_out, x):
    #     # aggr_out has shape [N, out_channels]

    #     # Step 5: Return new node embeddings.
    #     return aggr_out + self.central_lin(x).relu()

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

class GCN_2_layers(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.conv1 = custom_GCNConv(self.input_size, self.hidden_size)
        self.conv2 = custom_GCNConv(self.hidden_size, self.hidden_size)
        self.conv3 = custom_GCNConv(self.hidden_size, self.hidden_size)
        self.conv4 = custom_GCNConv(self.hidden_size, self.output_size)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.conv3(x, edge_index, edge_attr)
        x = self.conv4(x, edge_index, edge_attr)
        x = torch.sigmoid(x)
        return x
#endregion
