import numpy as np
import networkx as nx
import copy
import cvxopt
from sklearn.metrics import roc_auc_score

class KernelSVC:
    def __init__(self, C, epsilon=1e-3):
        self.scaleC = C
        self.alpha = None
        self.epsilon = epsilon

    def fit(self, K, y, class_weights=None):
        """Fit the support vector classifier.

        Args:
            K (np.ndarray): N*N Training Gram Matrix.
            y (np.ndarray): N Training labels.
            class_weights (dict, optional): Dictionnary of format {-1:w1, 1:w2}, C-weights for each label class. Defaults to None.
        """
        N = len(y)
        y = y*2-1 # rescaling values to -1, 1
        self.y = y

        assert (np.unique(y) == np.array([-1, 1])).all(), print('y must take values in [-1, 1]')

        # Specific C for each label:
        if type(class_weights) == dict:
            C = class_weights[-1] * (self.y == -1) + class_weights[1] * (self.y == 1)
        else:
            C = np.ones(N)
        C = self.scaleC * C

        # Quadratic objective
        P = np.diag(self.y) @ K @ np.diag(self.y)
        q = -np.ones(N)

        # Constraints
        G = np.kron(np.array([[-1.0], [1.0]]), np.eye(N))
        h = np.kron(np.array([0.0, 1.0]), C)
        A = self.y.reshape(1, -1).astype("float")
        b = np.array([[0.0]]).astype("float")

        # Optimization
        out = cvxopt.solvers.qp(
            P=cvxopt.matrix(P),
            q=cvxopt.matrix(q),
            G=cvxopt.matrix(G),
            h=cvxopt.matrix(h),
            A=cvxopt.matrix(A),
            b=cvxopt.matrix(b),
        )

        # Alpha
        self.alpha = np.array(out["x"]).reshape((N,))

        # Margin Points
        support_idx = np.where((self.alpha < C - self.epsilon) & (self.alpha > self.epsilon))

        # Offset for the classifier
        self.b = np.median(y[support_idx] - K[support_idx]@np.multiply(self.alpha, self.y))

    def predict(self, K):
        """Prediction for the fitted SVC classifier.

        Args:
            K (numpy.ndarray): M*N Gram Matrix.

        Returns:
            numpy.ndarray: M prediction logits.
        """
        return K@np.multiply(self.alpha, self.y) + self.b

def stratified_cross_val(
        G_train,
        train_labels,
        G_test,
        C,
        class_weights,
        n_fold=6,
        seed=42,
        verbose=False
    ):
    """Run a stratified cross validation on training Gram Matrix G_train and use resulting models to make predictions on
    test Gram Matrix G_test.

    Args:
        G_train (numpy.ndarray): N*N training Gram Matrix.
        train_labels (numpy.ndarray): N training labels.
        G_test (numpy.ndarray): M*N Testing Gram Matrix.
        C (float): C.
        class_weights (dict): weights for positive and negative class to apply to C format : {-1:w1, 1:w2}.
        n_fold (int, optional): Number of folds for cross val. Defaults to 6.
        seed (int, optional): Seed for reproducibility. Defaults to 42.
        verbose (bool, optional): Wether to display scores during cross val. Defaults to False.

    Returns:
        list(KernelSVC), list(floats), list(numpy.ndarray): cross val models, scores and predictions.
    """
    indexes_train = np.arange(len(train_labels))
    indexes_pos = np.where(train_labels == 1)[0]
    indexes_neg = np.where(train_labels == 0)[0]

    np.random.seed(seed)
    np.random.shuffle(indexes_pos)
    np.random.shuffle(indexes_neg)

    folds_pos = np.array_split(indexes_pos, n_fold)
    folds_neg = np.array_split(indexes_neg, n_fold)

    models = []
    scores = []
    test_preds = []

    for i in range(n_fold):
        if verbose:
            print(f'##### {n_fold}-FOLD CROSS VAL: starting fold {i+1} #####')

        val_idx = np.concatenate((folds_pos[i], folds_neg[i]))
        train_idx = list(set(indexes_train) - set(val_idx))

        G_train_fold = G_train[train_idx][:,train_idx]
        y_train_fold = train_labels[train_idx]

        if verbose:
            print(f'Percentage of positive values: {np.sum(y_train_fold)/len(y_train_fold):%}%')

        G_val_fold = G_train[val_idx][:,train_idx]
        y_val_fold = train_labels[val_idx]

        G_test_fold = G_test[:,train_idx]

        model = KernelSVC(C=C)
        model.fit(G_train_fold, y_train_fold, class_weights=class_weights)

        y_val_pred = model.predict(G_val_fold)
        val_score = roc_auc_score(y_val_fold, y_val_pred)

        if verbose:
            print(f'Val score: {val_score}')

        y_test_pred = model.predict(G_test_fold)

        models.append(model)
        scores.append(val_score)
        test_preds.append(y_test_pred)

    return models, scores, test_preds

def get_all_atom_types(G) :

    """ 
    Compute the labels of all the atoms from a graph.

    Parameters: 
    G (nx.Graph): input graph

    Returns:
    atom_types (dict): with (key,value) = (atom_id, atom_label)
    """
    atom_types = {}
    for k in G.nodes :
        atom_types[k] = G.nodes[k]['labels'][0]
    return atom_types

def get_atom_type(G, vertex_id) :

    """ 
    Compute the label of one atom from a graph.

    Parameters: 
    G (nx.Graph): Input graph
    vertex_id (int): atom's ID

    Returns:
    label (int): atom's label
    """

    label = G.nodes[vertex_id]['labels'][0] 
    return label

def get_neighbors_id(G, vertex_id) :

    """ 
    Compute the list of atom's neighbors in a graph.

    Parameters:
    G (nx.Graph): input graph
    vertex_id (int): atom's ID

    Returns:
    L (list): list of the IDs of the atom's neighbors 
    """

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

def kNN(G_val, y_train, k=3):
    """
    Given a feature dictionnary whose label is to be predicted, the WL feature vector of the graphs in the training set 
    and the number of nearest neighbors k, returns the predicted label of G.
    """
    neighbours = np.argpartition(G_val, k, axis=1)[:,:k]

    y_pred = []
    for i in range(len(neighbours)):
        is_positive = sum([y_train[i] for i in neighbours[i]])
        if is_positive > k/2 :
            y_pred.append(1)
        else:
            y_pred.append(0)
    return np.array(y_pred)

def WL_features(Graph, max_iter, verbose=False):
    """
    Implements the Weisfeiler-Lehman algorithm to compute a dictionnary vector
    describing a graph in input.

    Parameters:
    G (nx.Graph): Input graph
    max_iter (int): Number of iterations of the algorithm (first iteration not included. max_iter = h).

    Returns:
    patterns (dict): Feature dictionnary of the input graph (keys : patterns represented as strings, values : number of occurences).
    """
    G = copy.deepcopy(Graph)
    
    # Create a dictionnary with labels of all the iterations (keys) and their count (values)
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
            
            # Identify neighbors of current vertex.
            # We sort them according to their label to make sure we don't have redundant patterns.
            neighbors_id = sorted_neighbors_id(G, vertex_id, pattern_dict)
            
            # Identify patterns and fill exhaustively pattern list (with potentially repetition)
            # The spaces and dashes added below enable a better by-hand verification of the WL transform.
            pattern = f'{vertex_type}' + iter*'-' + '>'
            for neighbor in neighbors_id :
                pattern += str(get_atom_type(G, neighbor))
                pattern += (iter-1)*' '
            
            pattern_type.append(pattern)

        for k in range(len(pattern_type)) :
            # Change vertex label (relabelling)
            G.nodes[list(G.nodes)[k]]['labels'][0] = pattern_type[k]

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

def WLK_linear(l1, l2):
    """Implements inner product for WL features dicts (linear kernel).

    Args:
        l1 (dict): Feature dict built with WL_features.
        l2 (dict): Feature dict built with WL_features.

    Returns:
        float: Inner product, value of the kernel.
    """
    common_patterns = l1.keys() & l2.keys()
    wkl = 0
    for pattern in common_patterns:
        wkl += l1[pattern]*l2[pattern]
    return wkl

def WLK_l2_norm(l1, l2):
    """Computes l2 norm between WL feature dict from WL_features. Used to compute the RBF Kernel.

    Args:
        l1 (dict): Feature dict built with WL_features.
        l2 (dict): Feature dict built with WL_features.

    Returns:
        float: l2 norm.
    """
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

def clean_dataset(dataset, labels, discard=True) :
    """
    Given a dataset of networkx graphs
    and its associated label list, 
    clean it by iterating over molecules having
    more than one connected component and by either :

    1. discard the molecule if this is actually two
    fully connected graphs. The function DOES NOT 
    discard molecules if `discard` is set to False.

    2. remove only single nodes from the graph if there happens to be
    isolated atoms.

    Parameters: 
    dataset (list): list of nx.Graph
    labels (np.array): array of graphs labels
    discard (boolean): whether to discard molecules
        if it contains two distinct fully connected molecules.
    
    Returns:
    cleaned_dataset (list): list of nx.Graph
    new_labels (np.array): array of the associated labels

    """
    cleaned_dataset = []
    new_labels = []

    for mol_id in range(len(dataset)) :

        # Case of disconnected graphs
        if len(list(nx.connected_components(dataset[mol_id]))) > 1 :

            # Case of isolated atoms
            if len(sorted(list(nx.connected_components(dataset[mol_id])), key=len)[-2]) == 1 :
                new_mol = copy.deepcopy(dataset[mol_id])
                new_mol.remove_nodes_from(list(nx.isolates(new_mol)))
                cleaned_dataset.append(new_mol)
                new_labels.append(labels[mol_id])
            
            # Keep molecule as is if it has two connected graphs
            elif not discard :
                cleaned_dataset.append(dataset[mol_id])
                new_labels.append(labels[mol_id])
        
        # Case of one clean connected graph 
        else :
            cleaned_dataset.append(dataset[mol_id])
            new_labels.append(labels[mol_id])


    return cleaned_dataset, np.array(new_labels)

def nth_order_walk(G1, G2, n=3):

    """ 
    Given two graphs, compute their n-th order walk kernel
    using spectral theorem.

    Parameters:
    G1 (nx.Graph): first graph
    G2 (nx.Graph): second graph
    n (int): Length of the walks considered

    Returns:
    Kernel evaluation (float)

    """

    prod_graph = nx.tensor_product(G1, G2)
    adj_matrix = nx.adjacency_matrix(prod_graph).toarray()
    N = adj_matrix.shape[0]

    # Using spectral theorem to compute A^n.
    eigenval, eigenvect = np.linalg.eig(adj_matrix)

    P = eigenvect
    P_inv = np.linalg.inv(P)

    D = np.zeros((N,N))
    for k in range(N) :
        D[k,k] = np.real(eigenval[k])**n

    An = P @ D @ P_inv

    return np.real(np.sum(An))