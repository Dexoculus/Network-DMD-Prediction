import numpy as np
import pandas as pd
from numpy.linalg import pinv, eig
from numpy.lib.stride_tricks import sliding_window_view


class DMDc:
    """
    Implementation of Dynamic Mode Decomposition with Control (DMDc) algorithm using Least Squares.
    """
    def category(self):
        """
        Returns which kind of DMD algorithm this class is.
        """
        return 'DMDc'

    def get_dmdc_pair(self, data, control, ts_length):
        """
        Generates data pair (X1, X2, U1, U2) for DMDc algorithm.

        Args:
            data (numpy.ndarray or pandas.DataFrame):
                Flattened, row-wise, time series input data. Shape: (num_nodes*num_nodes, num_samples)
            control (numpy.ndarray or pandas.DataFrame):
                Control input data corresponding to each time step. Shape: (num_nodes*embedding_dims, num_samples)
            ts_length (int):
                Length of time series window.
        """
        # Validate and preprocess input data
        if isinstance(data, pd.DataFrame):
            y = data.values.T  # Shape: (feature_dim, num_samples)
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                y = data.reshape(1, -1)  # Shape: (1, num_samples)
            elif data.ndim == 2:
                y = data.T  # Shape: (feature_dim, num_samples)
            else:
                raise ValueError("data must be 1 or 2 dimensional numpy ndarray.")
        else:
            raise TypeError("data must be pandas DataFrame or numpy ndarray.")

        if isinstance(control, pd.DataFrame):
            u = control.values.T  # Shape: (control_dim, num_samples)
        elif isinstance(control, np.ndarray):
            if control.ndim == 1:
                u = control.reshape(1, -1)  # Shape: (1, num_samples)
            elif control.ndim == 2:
                u = control.T  # Shape: (control_dim, num_samples)
            else:
                raise ValueError("control must be 1 or 2 dimensional numpy ndarray.")
        else:
            raise TypeError("control must be pandas DataFrame or numpy ndarray.")

        num_features, num_samples = y.shape
        control_dim, control_samples = u.shape

        if num_samples != control_samples:
            raise ValueError("data and control must have the same number of samples.")

        if num_samples < ts_length + 1:
            raise ValueError("ts_length must be less than or equal to the number of data samples minus one.")

        # Generate sliding windows for data and control
        y_windows = sliding_window_view(y, window_shape=ts_length, axis=1)
        u_windows = sliding_window_view(u, window_shape=ts_length, axis=1)

        # Reshape to (num_features * ts_length, num_windows)
        tensor_y = y_windows.reshape(num_features * ts_length, -1)
        tensor_u = u_windows.reshape(control_dim * ts_length, -1)

        # Generate X1, X2, U1, U2
        self.X1 = tensor_y[:, :-1]      # Shape: (m, n-1)
        self.X2 = tensor_y[:, 1:]       # Shape: (m, n-1)
        self.U1 = tensor_u[:, :-1]      # Shape: (c, n-1)
        self.U2 = tensor_u[:, 1:]       # Shape: (c, n-1)
        self.x0 = self.X1[:, 0]         # Initial state vector

    def fit(self, data, control, ts_length, r=None, mode_selection=False, lambda_min=0.9, lambda_max=1.1):
        """
        Fit the DMDc model using Least Squares.

        Args:
            data (numpy.ndarray or pandas.DataFrame): Input state data.
            control (numpy.ndarray or pandas.DataFrame): Input control data.
            ts_length (int): Length of time series window.
            r (int, optional): Rank for potential future use (not used in Least Squares approach).
            mode_selection (bool, optional): Whether to perform mode selection. Default is False.
            lambda_min (float, optional): Minimum threshold for eigenvalue magnitude. Used if mode_selection is True.
            lambda_max (float, optional): Maximum threshold for eigenvalue magnitude. Used if mode_selection is True.
        """
        if ts_length <= 0:
            raise ValueError("ts_length must be a positive integer.")
        self.ts_length = ts_length

        # Generate data pair
        self.get_dmdc_pair(data=data, control=control, ts_length=ts_length)

        # Perform Least Squares to compute [A | B]
        # Concatenate X1 and U1 vertically: Shape (m + c, n-1)
        Y = np.vstack([self.X1, self.U1])  # Shape: (m + c, n-1)
        print(f"Performing Least Squares with Y shape: {Y.shape} and X2 shape: {self.X2.shape}")

        # Compute pseudo-inverse of Y
        Y_pinv = pinv(Y)  # Shape: (n-1, m + c)
        print(f"Pseudo-inverse of Y shape: {Y_pinv.shape}")

        # Compute [A | B] = X2 @ Y_pinv
        A_B = self.X2 @ Y_pinv  # Shape: (m, m + c)
        print(f"[A | B] shape: {A_B.shape}")  # Expected: (49, 98)

        # Split [A | B] into A and B
        m = self.X1.shape[0]  # 49
        c = self.U1.shape[0]  # 49

        self.A = A_B[:, :m]     # Shape: (49, 49)
        self.B = A_B[:, m:]     # Shape: (49, 49)

        print(f"A shape: {self.A.shape}")  # Expected: (49,49)
        print(f"B shape: {self.B.shape}")  # Expected: (49,49)

        # Get eigenvalues and eigenvectors of A
        self.get_eig_A()

        if mode_selection:
            # Perform mode selection based on eigenvalue magnitudes
            self.select_modes(lambda_min=lambda_min, lambda_max=lambda_max)

    def get_eig_A(self):
        """
        Calculate the eigenvalues and eigenvectors of matrix A.
        """
        print("Performing eigendecomposition on A")
        self.lamb, self.w = eig(self.A)
        print(f"Eigenvalues shape: {self.lamb.shape}")       # Expected: (49,)
        print(f"Eigenvectors shape: {self.w.shape}")       # Expected: (49, 49)

    def select_modes(self, lambda_min=0.9, lambda_max=1.1):
        """
        Select Important DMDc modes based on the magnitude of eigenvalues.

        Args:
            lambda_min (float): Minimum threshold for eigenvalue magnitude.
            lambda_max (float): Maximum threshold for eigenvalue magnitude.
        """
        abs_lamb = np.abs(self.lamb)
        idx = np.where((abs_lamb >= lambda_min) & (abs_lamb <= lambda_max))[0]

        print(f"Selecting modes with eigenvalue magnitudes between {lambda_min} and {lambda_max}")
        print(f"Selected mode indices: {idx}")

        # Update with selected eigenvalues and eigenvectors
        self.lamb = self.lamb[idx]
        self.w = self.w[:, idx]

        print(f"Selected eigenvalues shape: {self.lamb.shape}")      # Expected: (num_selected_modes,)
        print(f"Selected eigenvectors shape: {self.w.shape}")      # Expected: (49, num_selected_modes)

    def predict_future(self, t: int, control_future=None):
        """
        Predicting System state at future time t using DMDc

        Args:
            t (int):
                Future time t to predict.
            control_future (numpy.ndarray, optional):
                Future control inputs up to time t. Shape should be (c, t).

        Returns:
            numpy.ndarray:
                Predicted system state. Shape: (m,)
        """
        if t <= 0:
            raise ValueError("t must be a positive integer.")

        print(f"Predicting future state at t+{t}")

        # Initialize prediction with A^t * x0
        x_t = np.linalg.matrix_power(self.A, t) @ self.x0  # Shape: (m,)

        print(f"x_t shape after A^t * x0: {x_t.shape}")  # Expected: (49,)

        # Initialize control effect
        control_effect = np.zeros_like(x_t, dtype=np.float64)  # Shape: (m,)

        if control_future is not None:
            # Validate control_future shape
            expected_shape = (self.B.shape[1], t)  # (c, t)
            if control_future.shape != expected_shape:
                raise ValueError(f"control_future must have shape {expected_shape}, but got {control_future.shape}")

            for k in range(t):
                # Influence of control input at time k
                u_k = control_future[:, k]  # Shape: (c,)
                control_effect += np.linalg.matrix_power(self.A, t - k -1) @ self.B @ u_k  # Shape: (m,)

                print(f"Control effect after processing step {k}: {control_effect.shape}")

        # Add control effect to the state prediction
        x_t += control_effect  # Shape: (m,)

        print(f"x_t shape after adding control effect: {x_t.shape}")  # Expected: (49,)

        return x_t