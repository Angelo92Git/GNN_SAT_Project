import torch
import numpy as np
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.utils import degree
from my_utils import formula_utils as F
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from copy import deepcopy

def convert_instance_to_VCG_with_meta_node(formula):
    """
    Convert a boolean formula instance to a heterogeneous graph as per NeuroBack by Wang et al.
    Polarity is encoded as an edge attribute.
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
 
    data = T.ToUndirected()(data)
    return data

def convert_instance_to_VCG(formula):
    """
    Convert a boolean formula instance to VCG* as per G4SatBench by Li et al.
    Polarity is encoded as an edge attribute.
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
 
    data = T.ToUndirected()(data)
    return data

def convert_instance_to_VCG_bi_with_meta_node(formula):
    """
    Convert a boolean formula instance to a heterogeneous graph as per NeuroBack by Wang et al.
    Polarity is encoded as a new relation.
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
 
    sources = np.array(sources)
    polarity = np.array(polarity)
    pos_sources = torch.from_numpy(sources[polarity]).long()
    neg_sources = torch.from_numpy(sources[~polarity]).long()
    targets = np.array(targets)
    pos_targets = torch.from_numpy(targets[polarity]).long()
    neg_targets = torch.from_numpy(targets[~polarity]).long()
    data['meta', 'connects', 'clause'].edge_index = torch.stack([torch.zeros(num_clauses).long(), torch.arange(num_clauses).long()], dim=0)
    data['clause', 'contains_pos', 'variable'].edge_index = torch.stack([pos_sources, pos_targets], dim=0)
    data['clause', 'contains_neg', 'variable'].edge_index = torch.stack([neg_sources, neg_targets], dim=0)
    
    sources = torch.from_numpy(sources).long()
    targets = torch.from_numpy(targets).long()
    edge_index = torch.stack([sources, targets], dim=0)
    src, trg = edge_index
    data['meta'].deg = num_clauses
    data['clause'].deg = degree(src)
    data['variable'].deg = degree(trg)
 
    data = T.ToUndirected()(data)
    return data

def convert_instance_to_VCG_bi(formula):
    """
    Convert a boolean formula instance to VCG* as per G4SatBench by Li et al.
    Polarity is encoded as a new relation.
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
 
    sources = np.array(sources)
    polarity = np.array(polarity)
    pos_sources = torch.from_numpy(sources[polarity]).long()
    neg_sources = torch.from_numpy(sources[~polarity]).long()
    targets = np.array(targets)
    pos_targets = torch.from_numpy(targets[polarity]).long()
    neg_targets = torch.from_numpy(targets[~polarity]).long()
    data['clause', 'contains_pos', 'variable'].edge_index = torch.stack([pos_sources, pos_targets], dim=0)
    data['clause', 'contains_neg', 'variable'].edge_index = torch.stack([neg_sources, neg_targets], dim=0)
    
    sources = torch.from_numpy(sources).long()
    targets = torch.from_numpy(targets).long()
    edge_index = torch.stack([sources, targets], dim=0)
    src, trg = edge_index
    data['clause'].deg = degree(src)
    data['variable'].deg = degree(trg)
 
    assert data.node_types == ['clause', 'variable']
    assert data.edge_types == [('clause','contains_pos','variable'), ('clause', 'contains_neg', 'variable')]
    assert data['clause'].num_nodes == num_clauses
    assert data['clause'].num_features == 1
    assert data['variable'].num_nodes == num_variables
    assert data['variable'].num_features == 1
    
    data = T.ToUndirected()(data)
    return data

def convert_instance_to_LCG_with_meta_node(formula, add_self_loops=False):
    """
    Convert a boolean formula instance to LCG* as per G4SatBench by Li et al.
    """
    #TODO create a clean node index for literals
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
            sources.append(clause_id)
            targets.append(literal)
    
    l2l_sources = []
    l2l_targets = []
    positive_literals = list(filter(lambda x: x>0, literals))
    negative_literals = list(filter(lambda x: x<0, literals))
    for literal in positive_literals:
        if -literal in negative_literals:
            l2l_sources.append(literal)
            l2l_targets.append(-literal)

    num_literals = len(literals)
    data = HeteroData()
    data.y = torch.tensor([[label]]).float()
    data['meta'].x = torch.tensor([[0]]).float()
    data['meta'].node_index = torch.tensor([0]).long()
    data['clause'].x = torch.from_numpy(-np.expand_dims(np.ones(num_clauses),1)).float()
    data['clause'].node_index = torch.arange(num_clauses).long()
    data['literal'].x = torch.from_numpy(np.expand_dims(np.ones(num_literals),1)).float()
    data['literal'].node_index = torch.arange(num_literals).long()
    data['literal'].literal = torch.from_numpy(np.array(literals)).long()

    literal_node_index_lookup = {literal.item(): node_index.item() for literal, node_index in zip(data['literal'].literal, data['literal'].node_index)}
    targets = [literal_node_index_lookup[literal] for literal in targets]
    l2l_sources = [literal_node_index_lookup[literal] for literal in l2l_sources]
    l2l_targets = [literal_node_index_lookup[literal] for literal in l2l_targets]
 
    sources = torch.from_numpy(np.array(sources)).long()
    targets = torch.from_numpy(np.array(targets)).long()
    l2l_sources = torch.from_numpy(np.array(l2l_sources)).long()
    l2l_targets = torch.from_numpy(np.array(l2l_targets)).long()
    data['clause', 'contains', 'literal'].edge_index = torch.stack([sources, targets], dim=0)
    data['meta', 'connects', 'clause'].edge_index = torch.stack([torch.zeros(num_clauses).long(), torch.arange(num_clauses).long()], dim=0)
    data['literal', 'paired_with', 'literal'].edge_index = torch.stack([l2l_sources, l2l_targets], dim=0)
    
    data = T.ToUndirected()(data)
    if add_self_loops:
        data = T.AddSelfLoops()(data)
    return data

def convert_instance_to_LCG(formula, add_self_loops=False):
    """
    Convert a boolean formula instance to LCG* as per G4SatBench by Li et al.
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
            sources.append(clause_id)
            targets.append(literal)
    
    l2l_sources = []
    l2l_targets = []
    positive_literals = list(filter(lambda x: x>0, literals))
    negative_literals = list(filter(lambda x: x<0, literals))
    for literal in positive_literals:
        if -literal in negative_literals:
            l2l_sources.append(literal)
            l2l_targets.append(-literal)


    num_literals = len(literals)
    data = HeteroData()
    data.y = torch.tensor([[label]]).float()
    data['clause'].x = torch.from_numpy(-np.expand_dims(np.ones(num_clauses),1)).float()
    data['clause'].node_index = torch.arange(num_clauses).long()
    data['literal'].x = torch.from_numpy(np.expand_dims(np.ones(num_literals),1)).float()
    data['literal'].node_index = torch.arange(num_literals).long()
    data['literal'].literal = torch.from_numpy(np.array(literals)).long()

    literal_node_index_lookup = {literal.item(): node_index.item() for literal, node_index in zip(data['literal'].literal, data['literal'].node_index)}
    targets = [literal_node_index_lookup[literal] for literal in targets]
    l2l_sources = [literal_node_index_lookup[literal] for literal in l2l_sources]
    l2l_targets = [literal_node_index_lookup[literal] for literal in l2l_targets]
 
    sources = torch.from_numpy(np.array(sources)).long()
    targets = torch.from_numpy(np.array(targets)).long()
    l2l_sources = torch.from_numpy(np.array(l2l_sources)).long()
    l2l_targets = torch.from_numpy(np.array(l2l_targets)).long()
    data['clause', 'contains', 'literal'].edge_index = torch.stack([sources, targets], dim=0)
    data['literal', 'paired_with', 'literal'].edge_index = torch.stack([l2l_sources, l2l_targets], dim=0)
    
    data = T.ToUndirected()(data)
    if add_self_loops:
        data = T.AddSelfLoops()(data)
    return data

def convert_hetero_to_homogeneous_with_self_loops(data):
    """
    Convert list of heterogeneous graph to a homogeneous graph.
    """
    data = data.to_homogeneous()
    data = T.AddSelfLoops(attr = "edge_attr", fill_value = 1.)(data)
    return data

def draw_VCG_graph_from_formula(formula):
    G = nx.Graph()
    num_vars = formula[1]
    num_clauses = len(formula[2])
    for clause_idx, clause in enumerate(formula[2]):
        for literal in clause:
            sign, v_idx = F.literal2v_idx(literal)
            G.add_edge(clause_idx + num_vars, v_idx, weight=sign)
    top_vars = range(num_vars)
    pos_edges = [(u,v) for u,v,d in G.edges(data=True) if d['weight'] == True]
    neg_edges = [(u,v) for u,v,d in G.edges(data=True) if d['weight'] == False]
    pos = nx.bipartite_layout(G, top_vars, align="horizontal")
    fig, ax = plt.subplots(figsize=(12,7))
    ax.set_title(f"Satisfiability: {formula[0]} (red indicates negative polarity)")
    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=pos_edges, edge_color='blue')
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=neg_edges, edge_color='red')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_labels(G, pos, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels)
    plt.show()
    return

def draw_VCG_graph_from_data(data):
    data = deepcopy(data)
    data = data.to_homogeneous()
    G = to_networkx(data, node_attrs=['node_index','x'], edge_attrs=['edge_attr'], to_undirected=True)
    top_var_cond = data.x.squeeze()==1
    top_vars = top_var_cond.nonzero().squeeze().numpy()
    pos_edges = [(u,v) for u,v,d in G.edges(data=True) if d['edge_attr'] == True]
    neg_edges = [(u,v) for u,v,d in G.edges(data=True) if d['edge_attr'] == False]
    pos = nx.bipartite_layout(G, top_vars, align="horizontal")
    fig, ax = plt.subplots(figsize=(12,7))
    ax.set_title(f"Satisfiability: {data.y.item()} (red indicates negative polarity)")
    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist = pos_edges, edge_color='blue')
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist = neg_edges, edge_color='red')
    edge_labels = nx.get_edge_attributes(G, 'edge_attr')
    edge_labels = {k: bool(v) for k, v in edge_labels.items()}
    nx.draw_networkx_labels(G, pos, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels)
    plt.show()
    return
