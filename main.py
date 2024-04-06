import torch
from torch.nn import BCELoss
from torch_geometric.nn import GCNConv, to_hetero
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx

import os
from tqdm import tqdm, trange
from itertools import product
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

from my_utils import formula_utils as F
from my_utils import graph_utils as f2g
import models as m

torch.manual_seed(42)
skip_read = 1

if not skip_read:

    train_difficulty_levels = ["easy", "medium"]
    train_problem_types = ["3-sat"] 
    train_splits = ["train"]
    satisfiability = ["sat", "unsat"]

    test_difficulty_levels = ["hard"]
    test_problem_types = ["3-sat"]
    test_splits = ["test"]

    paths_to_training_instances = F.get_paths_to_instances(train_difficulty_levels, train_problem_types, train_splits, satisfiability, "./dataset")
    paths_to_testing_instances = F.get_paths_to_instances(test_difficulty_levels, test_problem_types, test_splits, satisfiability, "./dataset")

    # print(paths_to_training_instances)
    # print(paths_to_testing_instances)

    # Read in data
    training_formulas = F.read_instances_from_paths(paths_to_training_instances)
    testing_formulas = F.read_instances_from_paths(paths_to_testing_instances)
    # print(f"satisfiability: {formulas[0][0]} vars: {formulas[0][1]} clauses: {len(formulas[0][2])}")

    # Process data
    training_dataset = [f2g.convert_instance_to_VCG_with_meta_node(formula) for formula in training_formulas]
    training_dataset = [f2g.convert_to_homogeneous(data) for data in training_dataset] # each data object here is a graph
    testing_dataset = [f2g.convert_instance_to_VCG_with_meta_node(formula) for formula in testing_formulas]
    testing_dataset = [f2g.convert_to_homogeneous(data) for data in testing_dataset]

    # Save data
    with open("training_dataset_h.pkl", "wb") as f:
        pkl.dump(training_dataset, f)
    with open("testing_dataset_h.pkl", "wb") as f:   
        pkl.dump(testing_dataset, f)

with open("training_dataset_h.pkl", "rb") as f:
    training_dataset = pkl.load(f)
with open("testing_dataset_h.pkl", "rb") as f:
    testing_dataset = pkl.load(f)


# Create DataLoaders
num_training_instances = len(training_dataset)
num_testing_instances = len(testing_dataset)
print(f"Number of training instances: {num_training_instances}")
print(f"Number of testing instances: {num_testing_instances}")
train_loader = DataLoader(training_dataset, batch_size=20, shuffle=True)
test_loader = DataLoader(testing_dataset, batch_size=20, shuffle=True)

# Define the model, optimizer, and loss function
# model = m.custom_GCNConv(1, 1)
model = m.GCN_2_layers(1, 8, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
train_criterion = BCELoss()
test_criterion = BCELoss(reduction='sum')

# Train the model
model.train()
loss_record = []
pbar = tqdm(range(5), desc="Starting")
for epoch in pbar:
    for batched_graphs in train_loader:
        optimizer.zero_grad()
        out = model(batched_graphs.x, batched_graphs.edge_index, batched_graphs.edge_attr)
        _, node_counts = torch.unique_consecutive(batched_graphs.batch, return_counts=True)
        meta_node_indices = torch.cumsum(node_counts, dim=0).roll(1)
        meta_node_indices[0] = 0
        out = out[meta_node_indices]
        loss = train_criterion(out, batched_graphs.y)
        with torch.no_grad():
            pbar.set_description(f"Loss: {loss.item():.2f}")
            loss_record.append(loss.item())
        loss.backward()
        optimizer.step()

# Test the model
model.eval()
with torch.no_grad():
    loss = 0
    for batched_graphs in tqdm(test_loader):
        out = model(batched_graphs.x, batched_graphs.edge_index, batched_graphs.edge_attr)
        _, node_counts = torch.unique_consecutive(batched_graphs.batch, return_counts=True)
        meta_node_indices = torch.cumsum(node_counts, dim=0).roll(1)
        meta_node_indices[0] = 0
        out = out[meta_node_indices]
        loss += (torch.round(out) - batched_graphs.y).abs().sum()
        print(loss.item())

    print(f"Test loss: {loss/num_testing_instances*100:.2f}%")







