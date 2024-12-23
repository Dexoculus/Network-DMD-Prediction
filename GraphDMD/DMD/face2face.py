import numpy as np
import pandas as pd

class Dataset:
    def __init__(self, path):
        self.df = pd.read_csv(path)
        # Remove Columns containing 'LAPTOP'
        self.df = self.df.loc[:, ~self.df.columns.str.contains('LAPTOP', case=False)]
        self.adj = self._get_adjacency_matrices(self.df)

    def _get_original_data(self):
        return self.df
    
    # Function to create an adjacency matrix from a single row = timestep
    def _to_adjacency_matrix(self, row):
        edge = round(np.sqrt(len(row)))
        # Extract only the columns related to the adjacency matrix, ignoring the 'TIME' column
        adjacency_matrix = row[1:].values.reshape((edge, edge))
        
        return adjacency_matrix
    
    def _get_adjacency_matrices(self, df):
        adj_matrices = [self._to_adjacency_matrix(df.iloc[i]) for i in range(len(df))]

        return adj_matrices
    
    def __getitem__(self, idx):
        return self.adj[idx]
    
    def __len__(self):
        return len(self.adj)
    
    @property
    def shape(self):
        num_samples = len(self.adj)
        if num_samples > 0:
            sample_shape = self.adj[0].shape
            return (num_samples, ) + sample_shape
        else:
            return (0, )
        
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
