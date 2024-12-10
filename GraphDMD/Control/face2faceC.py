import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from sklearn.preprocessing import StandardScaler

class Dataset:
    """
    A Class for Loading and Preprocessing Dynamic Face-to-Face Interaction Networks dataset.
    Utilizes node2vec embeddings for all nodes as control inputs for each timestep.

    Attributes:
        df (pandas.DataFrame): The original dataframe after preprocessing.
        adj (list of numpy.ndarray): List of adjacency matrices for each timestep.
        node2vec_dimensions (int): Dimensionality of node2vec embeddings.
        embeddings_per_timestep (list of dict): List of node embeddings for each timestep.
        control (numpy.ndarray): Array of control input vectors for each timestep.
        scaler_data (StandardScaler): Scaler for state data.
        scaler_control (StandardScaler): Scaler for control inputs.
    """
    def __init__(self, path, walk_length=30, num_walks=200, workers=4, p=1, q=1):
        """
        Initialize the Dataset.

        Args:
            path (str): Path to the CSV file containing the dataset.
            node2vec_dimensions (int, optional): Dimensionality of node2vec embeddings.
                                                If None, set to number of nodes.
            walk_length (int, optional): Length of each random walk. Default is 30.
            num_walks (int, optional): Number of walks per node. Default is 200.
            workers (int, optional): Number of worker threads for node2vec. Default is 4.
            p (float, optional): Return parameter for node2vec. Default is 1.
            q (float, optional): Inout parameter for node2vec. Default is 1.
        """
        print("Loading Data...")
        self.df = pd.read_csv(path)
        # Remove Columns containing 'LAPTOP'
        self.df = self.df.loc[:, ~self.df.columns.str.contains('LAPTOP', case=False)]
        self.adj = self._get_adjacency_matrices(self.df)
        
        # Determine number of nodes from the first adjacency matrix
        if len(self.adj) == 0:
            raise ValueError("No adjacency matrices found in the dataset.")
        self.num_nodes = self.adj[0].shape[0]

        self.node2vec_dimensions = self.num_nodes  # Set to number of nodes

        # Generate node2vec embeddings for each timestep
        self.embeddings_per_timestep = self._generate_node2vec_embeddings_per_timestep(
            walk_length, num_walks, workers, p, q)
        
        # Generate control inputs by flattening all node embeddings
        self.control = self._get_control_inputs()
        
        # Normalize state and control data
        self.scaler_data, self.scaler_control = self._normalize_data()
        print("Finished Loading Data.")

    def _get_original_data(self):
        """
        Retrieve the original DataFrame.

        Returns:
            pandas.DataFrame: Original DataFrame after preprocessing.
        """
        return self.df

    def _to_adjacency_matrix(self, row):
        """
        Convert a DataFrame row to an adjacency matrix.

        Args:
            row (pandas.Series): A single row from the DataFrame.

        Returns:
            numpy.ndarray: Adjacency matrix of shape (num_nodes, num_nodes).
        """
        # Extract only the columns related to the adjacency matrix, ignoring 'TIME' and 'CONTROL_*' columns
        non_adj_columns = ['TIME'] + [col for col in self.df.columns if col.startswith('CONTROL')]
        adj_data = row.drop(non_adj_columns).values
        edge_length = int(np.sqrt(len(adj_data)))
        if edge_length ** 2 != len(adj_data):
            raise ValueError("Adjacency matrix data length is not a perfect square.")
        adjacency_matrix = adj_data.reshape((edge_length, edge_length))
        return adjacency_matrix

    def _get_adjacency_matrices(self, df):
        """
        Convert the entire DataFrame to a list of adjacency matrices.

        Args:
            df (pandas.DataFrame): The DataFrame containing adjacency data.

        Returns:
            list of numpy.ndarray: List of adjacency matrices for each timestep.
        """
        adj_matrices = [self._to_adjacency_matrix(df.iloc[i]) for i in range(len(df))]
        return adj_matrices

    def _generate_node2vec_embeddings_per_timestep(self, walk_length, num_walks, workers, p, q):
        """
        Generate node2vec embeddings for each timestep's adjacency matrix.

        Args:
            walk_length (int): Length of each random walk.
            num_walks (int): Number of walks per node.
            workers (int): Number of worker threads.
            p (float): Return parameter.
            q (float): Inout parameter.

        Returns:
            list of dict: List containing embedding dictionaries for each timestep.
        """
        embeddings_per_timestep = []
        for idx, adj_matrix in enumerate(self.adj):
            G = nx.from_numpy_array(adj_matrix)
            # Ensure the graph is connected; node2vec works better on connected graphs
            if not nx.is_connected(G):
                largest_cc = max(nx.connected_components(G), key=len)
                G = G.subgraph(largest_cc).copy()
                print(f"Time step {idx}: Graph is not connected. Using largest connected component with {len(G)} nodes.")
            try:
                node2vec = Node2Vec(
                    G, 
                    dimensions=7, 
                    walk_length=walk_length, 
                    num_walks=num_walks, 
                    workers=workers, 
                    p=p, 
                    q=q, 
                    quiet=True
                )
                model = node2vec.fit(window=3, min_count=1, batch_words=4)
                # Mapping from node index to embedding vector
                embeddings = {int(node): model.wv[str(node)] for node in G.nodes()}
                # For nodes not in the largest connected component, assign zero vectors
                all_nodes = set(range(self.num_nodes))
                missing_nodes = all_nodes - set(embeddings.keys())
                for node in missing_nodes:
                    embeddings[node] = np.zeros(self.node2vec_dimensions)
                embeddings_per_timestep.append(embeddings)
                print(f"{idx}th node2vec embedding processed.")
            except Exception as e:
                print(f"Time step {idx}: node2vec embedding failed with error: {e}")
                # Assign zero vectors if node2vec fails
                embeddings = {node: np.zeros(self.node2vec_dimensions) for node in range(self.num_nodes)}
                embeddings_per_timestep.append(embeddings)
        return embeddings_per_timestep

    def _get_control_inputs(self):
        """
        Generate control inputs by concatenating all node embeddings for each timestep.

        Returns:
            numpy.ndarray: Array of control input vectors for each timestep.
                           Shape: (num_samples, num_nodes * embedding_dimensions)
        """
        control = []
        for embeddings in self.embeddings_per_timestep:
            # Retrieve embeddings for all nodes
            selected_embeddings = [embeddings[node] for node in range(self.num_nodes)]
            # Concatenate all embeddings to form control input vector
            control_input = np.concatenate(selected_embeddings)
            control.append(control_input)
        control = np.array(control)  # Shape: (num_samples, num_nodes * embedding_dimensions)
        return control

    def _normalize_data(self):
        """
        Normalize the state and control data using StandardScaler.

        Returns:
            tuple: Scalers for data and control inputs.
        """
        scaler_data = StandardScaler()
        scaler_control = StandardScaler()
        # Flatten state vectors
        data = np.array([adj.flatten(order='C') for adj in self.adj])
        # Normalize
        data_scaled = scaler_data.fit_transform(data)
        control_scaled = scaler_control.fit_transform(self.control)
        # Update adjacency matrices with scaled data
        adj_scaled = [data_scaled[i].reshape((self.num_nodes, self.num_nodes)) for i in range(len(data_scaled))]
        self.adj = adj_scaled
        # Update control inputs with scaled data
        self.control = control_scaled
        return scaler_data, scaler_control

    def __getitem__(self, idx):
        """
        Retrieve the state vector and control input at the given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (state_vector, control_input)
                   state_vector: numpy.ndarray of shape (num_nodes * num_nodes,)
                   control_input: numpy.ndarray of shape (num_nodes * embedding_dimensions,)
        """
        state_matrix = self.adj[idx]
        state_vector = state_matrix.flatten(order='C')
        control_input = self.control[idx]
        return state_vector, control_input

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of adjacency matrices in the dataset.
        """
        return len(self.adj)

    @property
    def shape(self):
        """
        Get the shape of the dataset.

        Returns:
            tuple: Shape of the dataset as (num_samples, num_nodes * num_nodes).
        """
        num_samples = len(self.adj)
        if num_samples > 0:
            sample_shape = self.adj[0].shape
            return (num_samples, ) + sample_shape
        else:
            return (0, )

    def get_control_inputs(self, idx):
        """
        Get the control input vector at the given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            numpy.ndarray: Control input vector at index idx.
                           Shape: (num_nodes * embedding_dimensions,)
        """
        return self.control[idx]

    def get_state_vector(self, idx):
        """
        Get the state vector at the given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            numpy.ndarray: State vector at index idx.
                           Shape: (num_nodes * num_nodes,)
        """
        state_matrix = self.adj[idx]
        state_vector = state_matrix.flatten(order='C')
        return state_vector