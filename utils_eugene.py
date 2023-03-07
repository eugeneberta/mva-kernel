import numpy as np
import networkx as nx

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
            patterns[pattern] += 1
        else:
            patterns[pattern] = 1
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
                step_patterns[pattern] += 1
            else:
                step_patterns[pattern] = 1
            
            node_features[i, n] = pattern
        
        if len(step_patterns) == num_patterns:
            return patterns
        else:
            num_patterns = len(step_patterns)
            patterns.update(step_patterns)

    return patterns

def WLK(l1, l2, h=5):
    common_patterns = l1.keys() & l2.keys()
    wkl = 0
    for pattern in common_patterns:
        wkl += l1[pattern]*l2[pattern]
    return wkl