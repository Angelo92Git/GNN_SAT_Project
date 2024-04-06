import torch
from torch.nn import Linear, Parameter
import torch.nn.functional as funcs
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import contains_self_loops
from torch_geometric.utils import to_networkx

from torch_geometric.nn import GATConv, Linear, to_hetero

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels, bias=False)
        self.conv2 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin2 = Linear(-1, hidden_channels, bias=False)
        self.conv3 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin3 = Linear(-1, hidden_channels, bias=False)
        self.conv4 = GATConv((-1, -1), out_channels, add_self_loops=False)
        self.lin4 = Linear(-1, out_channels, bias=False)

    def forward(self, x, edge_index, edge_attr):
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
