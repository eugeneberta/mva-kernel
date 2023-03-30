import numpy as np
import networkx as nx
import copy

def get_all_atom_types(G) :
    atom_types = {}
    for k in G.nodes :
        atom_types[k] = G.nodes[k]['labels'][0]
    return atom_types

def weisfeiler_lehman(G, h):
    """
    Implements the Weisfeiler-Lehman algorithm to compute a feature vector
    describing a graph in input.
    
    Parameters:
    G (nx.Graph): Input graph
    h (int): Number of iterations of the algorithm
    
    Returns:
    patterns (dict): Feature vector of the input graph
    """
    N = G.number_of_nodes()
    
    # Initialize feature vectors
    node_features = np.zeros((h, N), dtype=object)
    
    patterns = {}
    for n in range(N):
        pattern = str(G.nodes[n]['labels'][0])
        if pattern in patterns:
            patterns[pattern] += 1/N
        else:
            patterns[pattern] = 1/N
        node_features[0, n] = pattern
    
    num_patterns = len(patterns)
    
    # Iterate h times
    for i in range(1, h):
        step_patterns = {}
        for n in range(N):
            neighbor_features = [node_features[i-1, n] for n in G.neighbors(n)]
            sorted_features = sorted(neighbor_features)
            pattern = str(node_features[i-1, n]) + '>' + ''.join([str(f) for f in sorted_features])

            if pattern in step_patterns:
                step_patterns[pattern] += (1+h)/N
            else:
                step_patterns[pattern] = (1+h)/N
            
            node_features[i, n] = pattern
        
        if len(step_patterns) == num_patterns:
            return patterns
        else:
            num_patterns = len(step_patterns)
            patterns.update(step_patterns)

    return patterns

def clean_dataset(dataset) :
    """
    Given a dataset of networkx graphs, clean it by iterating over molecules having
    more than one connected component and by either :

    1. discard the molecule if this is actually two
    fuly connected graphs
    2. remove only single nodes from the graph if there happens to be
    isolated atoms.
    
    Returns the new dataset.

    """
    cleaned_dataset = []

    for mol_id in range(len(dataset)) :

        # Case of disconnected graph
        if len(list(nx.connected_components(dataset[mol_id]))) > 1 :

            # Case of isolated atoms
            if len(sorted(list(nx.connected_components(dataset[mol_id])), key=len)[-2]) == 1 :
                
                new_mol = copy.deepcopy(dataset[mol_id])
                new_mol.remove_nodes_from(list(nx.isolates(new_mol)))
                cleaned_dataset.append(new_mol)
            
        else :
            cleaned_dataset.append(dataset[mol_id])

    return cleaned_dataset
