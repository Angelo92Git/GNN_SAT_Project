import torch
import numpy as np
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from my_utils import formula_utils as F

def convert_instance_to_VCG_with_meta_node(formula):
    """
    Convert a boolean formula instance to a heterogeneous graph as per NeuroBack by Wang et al.
    """
    label = formula[0]
    num_variables = formula[1]
    num_clauses = len(formula[2])
  
    sources = []
    targets = []
    polarity = []
    for clause_id, clause in enumerate(formula[2]):
        for literal in clause:
            sign, variable_idx = F.literal2v_idx(literal)
            sources.append(clause_id)
            targets.append(variable_idx)
            polarity.append(sign)
 
    data = HeteroData()
    data.y = torch.tensor([[label]]).float()
    data['meta'].x = torch.tensor([[0]]).float()
    data['meta'].node_index = torch.tensor([0]).long()
    data['clause'].x = torch.from_numpy(-np.expand_dims(np.ones(num_clauses),1)).float()
    data['clause'].node_index = torch.arange(num_clauses).long()
    data['variable'].x = torch.from_numpy(np.expand_dims(np.ones(num_variables),1)).float()
    data['variable'].node_index = torch.arange(num_variables).long()
 
    sources = torch.from_numpy(np.array(sources)).long()
    targets = torch.from_numpy(np.array(targets)).long()
    data['clause', 'contains', 'variable'].edge_index = torch.stack([sources, targets], dim=0)
    data['clause', 'contains', 'variable'].edge_attr = torch.from_numpy(np.array(polarity)).float()
    data['meta', 'connects', 'clause'].edge_index = torch.stack([torch.zeros(num_clauses).long(), torch.arange(num_clauses).long()], dim=0)
    data['meta', 'connects', 'clause'].edge_attr = torch.zeros(num_clauses).float()
 
    assert data.node_types == ['meta', 'clause', 'variable']
    assert data.edge_types == [('clause','contains','variable'), ('meta','connects','clause')]
    assert data['meta', 'connects', 'clause'].num_edges == num_clauses
    assert data['meta'].num_nodes == 1
    assert data['meta'].num_features == 1
    assert data['clause'].num_nodes == num_clauses
    assert data['clause'].num_features == 1
    assert data['variable'].num_nodes == num_variables
    assert data['variable'].num_features == 1
    
    data = T.ToUndirected()(data)
    return data

def convert_instance_to_VCG(formula):
    """
    Convert a boolean formula instance to a heterogeneous graph as per NeuroBack by Wang et al.
    """
    label = formula[0]
    num_variables = formula[1]
    num_clauses = len(formula[2])
  
    sources = []
    targets = []
    polarity = []
    for clause_id, clause in enumerate(formula[2]):
        for literal in clause:
            sign, variable_idx = F.literal2v_idx(literal)
            sources.append(clause_id)
            targets.append(variable_idx)
            polarity.append(sign)
 
    data = HeteroData()
    data.y = torch.tensor([[label]]).float()
    data['clause'].x = torch.from_numpy(-np.expand_dims(np.ones(num_clauses),1)).float()
    data['clause'].node_index = torch.arange(num_clauses).long()
    data['variable'].x = torch.from_numpy(np.expand_dims(np.ones(num_variables),1)).float()
    data['variable'].node_index = torch.arange(num_variables).long()
 
    sources = torch.from_numpy(np.array(sources)).long()
    targets = torch.from_numpy(np.array(targets)).long()
    data['clause', 'contains', 'variable'].edge_index = torch.stack([sources, targets], dim=0)
    data['clause', 'contains', 'variable'].edge_attr = torch.from_numpy(np.array(polarity)).float()
 
    assert data.node_types == ['clause', 'variable']
    assert data.edge_types == [('clause','contains','variable')]
    assert data['clause'].num_nodes == num_clauses
    assert data['clause'].num_features == 1
    assert data['variable'].num_nodes == num_variables
    assert data['variable'].num_features == 1
    
    data = T.ToUndirected()(data)
    return data

def convert_instance_to_LCG_with_meta_node(formula, add_self_loops=True):
    """
    Convert a boolean formula instance to a heterogeneous graph as per NeuroBack by Wang et al.
    """
    label = formula[0]
    num_variables = formula[1]
    num_clauses = len(formula[2])
  
    sources = []
    targets = []
    literals = []
    for clause_id, clause in enumerate(formula[2]):
        for literal in clause:
            if literal not in literals:
                literals.append(literal)
            
            literal_idx = F.literal2l_idx(literal)
            sources.append(clause_id)
            targets.append(literal_idx)
    
    l2l_sources = []
    l2l_targets = []
    positive_literals = list(filter(lambda x: x>0, literals))
    negative_literals = list(filter(lambda x: x<0, literals))
    for literal in positive_literals:
        if -literal in negative_literals:
            positive_literal_idx = F.literal2l_idx(literal)
            negative_literal_idx = F.literal2l_idx(-literal)
            l2l_sources.append(positive_literal_idx)
            l2l_targets.append(negative_literal_idx)


    num_literals = len(literals)
    data = HeteroData()
    data.y = torch.tensor([[label]]).float()
    data['meta'].x = torch.tensor([[0]]).float()
    data['meta'].node_index = torch.tensor([0]).long()
    data['clause'].x = torch.from_numpy(-np.expand_dims(np.ones(num_clauses),1)).float()
    data['clause'].node_index = torch.arange(num_clauses).long()
    data['literal'].x = torch.from_numpy(np.expand_dims(np.ones(num_literals),1)).float()
    data['literal'].node_index = torch.tensor(list(map(F.literal2l_idx, literals))).long()
 
    sources = torch.from_numpy(np.array(sources)).long()
    targets = torch.from_numpy(np.array(targets)).long()
    l2l_sources = torch.from_numpy(np.array(l2l_sources)).long()
    l2l_targets = torch.from_numpy(np.array(l2l_targets)).long()
    data['clause', 'contains', 'literal'].edge_index = torch.stack([sources, targets], dim=0)
    data['meta', 'connects', 'clause'].edge_index = torch.stack([torch.zeros(num_clauses).long(), torch.arange(num_clauses).long()], dim=0)
    data['literal', 'paired_with', 'literal'].edge_index = torch.stack([l2l_sources, l2l_targets], dim=0)
    assert data.node_types == ['meta', 'clause', 'literal']
    assert data.edge_types == [('clause','contains','literal'), ('meta','connects','clause'), ('literal','paired_with','literal')]
    assert data['meta', 'connects', 'clause'].num_edges == num_clauses
    assert data['meta'].num_nodes == 1
    assert data['meta'].num_features == 1
    assert data['clause'].num_nodes == num_clauses
    assert data['clause'].num_features == 1
    assert data['literal'].num_nodes == num_literals
    assert data['literal'].num_features == 1
    
    data = T.ToUndirected()(data)
    if add_self_loops:
        data = T.AddSelfLoops()(data)
    return data

def convert_instance_to_LCG(formula, add_self_loops=True):
    """
    Convert a boolean formula instance to a heterogeneous graph as per NeuroBack by Wang et al.
    """
    label = formula[0]
    num_variables = formula[1]
    num_clauses = len(formula[2])
  
    sources = []
    targets = []
    literals = []
    for clause_id, clause in enumerate(formula[2]):
        for literal in clause:
            if literal not in literals:
                literals.append(literal)
            
            literal_idx = F.literal2l_idx(literal)
            sources.append(clause_id)
            targets.append(literal_idx)
    
    l2l_sources = []
    l2l_targets = []
    positive_literals = list(filter(lambda x: x>0, literals))
    negative_literals = list(filter(lambda x: x<0, literals))
    for literal in positive_literals:
        if -literal in negative_literals:
            positive_literal_idx = F.literal2l_idx(literal)
            negative_literal_idx = F.literal2l_idx(-literal)
            l2l_sources.append(positive_literal_idx)
            l2l_targets.append(negative_literal_idx)


    num_literals = len(literals)
    data = HeteroData()
    data.y = torch.tensor([[label]]).float()
    data['clause'].x = torch.from_numpy(-np.expand_dims(np.ones(num_clauses),1)).float()
    data['clause'].node_index = torch.arange(num_clauses).long()
    data['literal'].x = torch.from_numpy(np.expand_dims(np.ones(num_literals),1)).float()
    data['literal'].node_index = torch.tensor(list(map(F.literal2l_idx, literals))).long()
 
    sources = torch.from_numpy(np.array(sources)).long()
    targets = torch.from_numpy(np.array(targets)).long()
    l2l_sources = torch.from_numpy(np.array(l2l_sources)).long()
    l2l_targets = torch.from_numpy(np.array(l2l_targets)).long()
    data['clause', 'contains', 'literal'].edge_index = torch.stack([sources, targets], dim=0)
    data['literal', 'paired_with', 'literal'].edge_index = torch.stack([l2l_sources, l2l_targets], dim=0)
    assert data.node_types == ['clause', 'literal']
    assert data.edge_types == [('clause','contains','literal'), ('literal','paired_with','literal')]
    assert data['clause'].num_nodes == num_clauses
    assert data['clause'].num_features == 1
    assert data['literal'].num_nodes == num_literals
    assert data['literal'].num_features == 1
    
    data = T.ToUndirected()(data)
    if add_self_loops:
        data = T.AddSelfLoops()(data)
    return data

def convert_to_homogeneous(data):
    """
    Convert list of heterogeneous graph to a homogeneous graph.
    """
    data = data.to_homogeneous()
    data = T.AddSelfLoops(attr = "edge_attr", fill_value = 1.)(data)
    return data

# TODO: Graph Plotting Utilities