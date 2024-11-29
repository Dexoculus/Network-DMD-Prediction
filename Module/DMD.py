import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from numpy.lib.stride_tricks import sliding_window_view

class DMD:
    """
    Implementation of Dynamic Mode Decomposition (DMD) algorithm.
    """
    def get_dmd_pair(self, data, ts_length: int):
        """
        Generates data pair (X1, X2) for DMD algorithm.

        Args:
            data (numpy.ndarray or pandas.DataFrame)
                Flatten, row, time series Input data.
            ts_length (int)
                Length of time series window.

        Raises:
            ValueError
                Accurs when ts_length is longer than number of data sample.
            TypeError
                Accurs when data type is not suppoted.
        """
        # Validation of input data type and shape
        if isinstance(data, pd.DataFrame): # pandas DataFrame
            y = data.values.T  # shape = (feature_dim, num_samples)
        elif isinstance(data, np.ndarray): # numpy array
            if data.ndim == 1:
                # Case of 1-iemensional array (e.g., (num_samples,))
                y = data.reshape(1, -1)  # shape = (1, num_samples)
            elif data.ndim == 2:
                # Case of 2-dimensional array (e.g., (num_samples, feature_dim))
                y = data.T  # shape = (feature_dim, num_samples)
            else:
                raise ValueError("data must be 1 or 2 diment numpy ndarray.")
        else:
            raise TypeError("data must be pandas DataFrame or numpy ndarray.")

        num_features, num_samples = y.shape  # extract number of features and samples

        if num_samples < ts_length: # When ts_length is longer than number of samples
            raise ValueError("ts_length must be less than or equal to the number of data samples.")

        # Generate time series slide window
        y_windows = sliding_window_view(y, window_shape=ts_length, axis=1)
        # y_windows.shape = (num_features, num_samples - ts_length + 1, ts_length)

        # (num_features * ts_length, num_windows)
        tensor = y_windows.reshape(num_features * ts_length, -1)

        # Generate X1, X2 
        self.x1 = tensor[:, :-1] # except last row from entire rows
        self.x2 = tensor[:, 1:]  # All rows except first row
        self.x0 = self.x1[:, 0]  # Initial state vector

    def fit(self, data, ts_length: int, r=None):
        """
        Fitting DMD Model 

        Args:
            data : numpy.ndarray or pandas.DataFrame
                Flatten, row, time series Input data.
            ts_length : int
                Length of time series window.

        Raises:
            ValueError
                Accurs when ts_length is not positive integer.
        """
        # Validates length of ts_length
        if ts_length <= 0:
            raise ValueError("ts_length must be positive integer.")
        self.ts_length = ts_length

        self.get_dmd_pair(data=data, ts_length=ts_length) # Generate data pair
        self.svd_x1(r) # Perform SVD on X1 and select rank
        self.get_atilde() # Calculate low-dimensional approximation matirx A~
        self.get_eig_atilde() # Get Eigen value and vector of A~        
        self.get_dmd() # Calculate DMD Mode matrix

    def svd_x1(self, r=None):
        """
        Performs SVD on the X1 matrix, and uses the selected rank r.

        Args:
            r (int, optional)
                Number of eigen values to calculate. If None, Use entire rank
        """
        # Performs SVD on the X1
        if r is not None and r < min(self.x1.shape):
            # Calculate r singular values using SVD
            u, s, vt = svds(self.x1, k=r)
            # Since svds returns singular values in ascending order, so sort them in descending order
            idx = np.argsort(s)[::-1]
            s = s[idx]
            u = u[:, idx]
            vt = vt[idx, :]
        else:
            # In case of using entire rank, apply numpy svd algorithm
            u, s, vt = np.linalg.svd(self.x1, full_matrices=False)

        # Save result of SVD
        self.u = u
        self.s = s
        self.vt = vt

    def get_atilde(self):
        """
        Calculate the approximate linear dynamics matrix A~ in low-dimensional space.
        """
        # Process if the denominator is zero when calculating the singular value reciprocal
        s_inv = np.array([1 / si if si > 1e-10 else 0 for si in self.s])
        # A~ = U^T * X2 * V * Σ^{-1}
        self.atilde = self.u.T @ self.x2 @ self.vt.T @ np.diag(s_inv)

    def get_eig_atilde(self):
        """
        Calculate the eigenvalues and eigenvectors of approximate matrix A~
        """
        self.lamb, self.w = np.linalg.eig(self.atilde)

    def get_dmd(self):
        """
        Calcultae DMD mode matrix \phi
        """
        # Process if the denominator is zero when calculating the singular value reciprocal
        s_inv = np.array([1 / si if si > 1e-10 else 0 for si in self.s])
        # Φ = X2 * V * Σ^{-1} * W
        self.phi = self.x2 @ self.vt.T @ np.diag(s_inv) @ self.w

    def predict_future(self, t: int):
        """
        Predicting System state at future time t

        Args:
        t (int)
            Future time t to predict.

        Returns:
            t_state.flatten() (numpy ndarray)
                Predicted system state. 
        """
        # Calculate coefficient of inital state DMD mode 
        pseudophix0 = np.linalg.pinv(self.phi) @ self.x0.reshape(-1, 1)
        atphi = self.phi @ np.diag(self.lamb ** t) # Calculate time
        xt = atphi @ pseudophix0 # Predict future state

        num_features = self.x0.shape[0] // self.ts_length
        t_state = xt[-num_features:]

        if np.all(np.abs(np.imag(t_state)) < 1e-6):
            return np.real(t_state).flatten() # Return real part if imaginary part is samll enough
        else:
            return t_state.flatten() # Return complex number