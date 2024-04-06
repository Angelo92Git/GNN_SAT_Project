import os
from itertools import product

# get paths to instances in dataset
def get_paths_to_instances(difficulty_levels, problem_types, splits, satisfiability, dataset_path):
    train_directory_paths = []
    for difficulty_level, problem_type, split, sat in product(difficulty_levels, problem_types, splits, satisfiability):
        directory_path = os.path.join(dataset_path, difficulty_level, problem_type, split, sat)
        entries = os.listdir(directory_path)
        file_count = sum(os.path.isfile(os.path.join(directory_path, entry)) for entry in entries)
        train_directory_paths.append((sat=='sat', file_count, directory_path))
    return train_directory_paths

# read in the instances from (satisfiability, file_count, path) tuples
def read_instances_from_paths(paths):
    formulas = []
    for sat, file_count, path in paths:
        for file in range(file_count):
            file_path = os.path.join(path, str(file).zfill(5)+".cnf")
            vars, clauses = parse_cnf_file(file_path)
            formulas.append((sat, vars, clauses))
    return formulas


# parse a file in DIMACS format
def parse_cnf_file(file_path, split_clauses=False):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        tokens = lines[i].strip().split()
        if len(tokens) < 1 or tokens[0] != 'p':
            i += 1
        else:
            break
    
    if i == len(lines):
        return 0, []
    
    header = lines[i].strip().split()
    n_vars = int(header[2])
    n_clauses = int(header[3])
    clauses = []
    learned = False

    if split_clauses:
        learned_clauses = []

    for line in lines[i+1:]:
        tokens = line.strip().split()
        if tokens[0] == 'c':
            if split_clauses and tokens[1] == 'augment':
                learned = True
            continue
        
        clause = [int(s) for s in tokens[:-1]]

        if not learned:
            clauses.append(clause)
        else:
            learned_clauses.append(clause)
    
    if not split_clauses:
        return n_vars, clauses
    else:
        return n_vars, clauses, learned_clauses

# transform literal to variable index (0 based)
def literal2v_idx(literal):
    assert abs(literal) > 0
    sign = literal > 0
    v_idx = abs(literal) - 1
    return sign, v_idx


# transform literal to literal index (0 based)
def literal2l_idx(literal):
    assert abs(literal) > 0
    sign = literal > 0
    v_idx = abs(literal) - 1
    if sign:
        return v_idx * 2
    else:
        return v_idx * 2 + 1