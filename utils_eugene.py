import numpy as np
import networkx as nx
import copy

def get_all_atom_types(G) :
    atom_types = {}
    for k in G.nodes :
        atom_types[k] = G.nodes[k]['labels'][0]
    return atom_types

def get_atom_type(G, vertex_id) :

    return G.nodes[vertex_id]['labels'][0]



def get_neighbors_id(G, vertex_id) :

    L = []
    neighbored_edges = [k for k in G.edges if vertex_id in k]

    for edge in neighbored_edges :

        if edge[0] != vertex_id :
            L.append(edge[0])
        else : 
            L.append(edge[1])

    return L


def sorted_neighbors_id(G, vertex_id, pattern_dict) :

    """ 
    Returns a dict {neighbors id: neighbors label} sorted by their label

    """
    # Define random but fixed order for labels
    unique_pattern_order = list(pattern_dict.keys())

    # Get a dict of neighbors_id with their label and order it according to the label order ahead
    neighbors_id = get_neighbors_id(G, vertex_id)
    neighbors_dict = {k:str(get_atom_type(G,k)) for k in neighbors_id}

    neighbors_dict_sorted = dict(sorted(neighbors_dict.items(), key=lambda item: unique_pattern_order.index(item[1])))

    return neighbors_dict_sorted


def WL(Graph, max_iter, verbose=False) :

    G = copy.deepcopy(Graph)

    
    feature_vector = {}

    # Initialize feature vector with initial labels
    atom_types = get_all_atom_types(G)
    atom_types_set = set(atom_types.values())

    pattern_dict = {str(pattern):list(atom_types.values()).count(pattern) for pattern in atom_types_set}
    old_pattern_dict_length = len(pattern_dict)
    feature_vector.update(pattern_dict)

    if verbose : 
        print('Pattern dict', pattern_dict)

    # Start iterations
    for iter in range(1, max_iter+1) :
        
        pattern_type = [] 

        for vertex_id in G.nodes :
            
            # Get vertex label
            vertex_type = get_atom_type(G, vertex_id)
            
            # Identify neighbors of current vertex
            neighbors_id = sorted_neighbors_id(G, vertex_id, pattern_dict)
            
            # Identify patterns and fill exhaustively pattern list (with potentially repetition)
            pattern = f'{vertex_type}' + iter*'-' + '>'
            for neighbor in neighbors_id :
                pattern += str(get_atom_type(G, neighbor))
                pattern += (iter-1)*' '
            
            pattern_type.append(pattern)

        for k in range(len(pattern_type)) :

            # Change vertex label (relabelling)
            G.nodes[k]['labels'][0] = pattern_type[k]

        # Count unique patterns  
        pattern_dict = {k:pattern_type.count(k) for k in set(pattern_type)}
        assert sum(pattern_dict.values()) == len(G.nodes)
        
  
        # Stop algorithm if iteration doesn't add information
        if len(pattern_dict) == old_pattern_dict_length :
            break
        else :
            old_pattern_dict_length = len(pattern_dict)
            # Update feature vector
            feature_vector.update(pattern_dict)  
        
        if verbose :
            print(f'Iteration {iter} finished !')
            print('Pattern dict', pattern_dict)
            print('Length of pattern dict', len(pattern_dict))
            print('Feature_vector', feature_vector)

    return feature_vector

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

def WLK_linear(l1, l2):
    common_patterns = l1.keys() & l2.keys()
    wkl = 0
    for pattern in common_patterns:
        wkl += l1[pattern]*l2[pattern]
    return wkl

def WLK_gaussian(l1, l2):
    s1 = set(l1.keys())
    s2 = set(l2.keys())
    common_patterns = list(s1 & s2)
    patterns1 = list(s1-s2)
    patterns2 = list(s2-s1)
    wkl = 0
    for pattern in common_patterns:
        wkl += (l1[pattern]-l2[pattern])**2
    for pattern in patterns1:
        wkl += l1[pattern]**2
    for pattern in patterns2:
        wkl += l2[pattern]**2
    return wkl

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
