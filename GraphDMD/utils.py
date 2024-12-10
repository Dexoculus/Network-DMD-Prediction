import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def DMD_touring(DMD_Class, dataset, ts_length,
                r=None,
                mode_selection=False,
                lambda_min=0.9,
                lambda_max=1.1):
    """
    Generalized function to perform DMD or DMDc on a given dataset.

    Args:
        DMD_Class (class): Class implementation of DMD or DMDc algorithm.
        dataset (class): Graph Dataset class.
        path (str): Path to the dataset.
        ts_length (int): Length of the time series window.
        r (int, optional): Rank for SVD truncation.
        mode_selection (bool, optional): Whether to perform mode selection. Default is False.
        lambda_min (float, optional): Minimum threshold for eigenvalue magnitude. Used if mode_selection is True.
        lambda_max (float, optional): Maximum threshold for eigenvalue magnitude. Used if mode_selection is True.
        control (bool, optional) : Whether DMDc or not. Default is False. 

    Returns:
        tuple: (Original data, Predicted data)
    """
    # Instantiate the DMD/DMDc model
    model = DMD_Class()
    
    # Prepare state and control inputs separately
    data = np.array([dataset.get_state_vector(i) for i in range(len(dataset))])      # Shape: (num_samples, m*m)

    # Fitting the model
    if model.category() == 'DMDc':
        control = np.array([dataset.get_control_inputs(i) for i in range(len(dataset))])  # Shape: (num_samples, m*m)
        model.fit(data=data,
                  control=control,
                  ts_length=ts_length,
                  r=r,
                  mode_selection=mode_selection,
                  lambda_min=lambda_min,
                  lambda_max=lambda_max)
    else:
        model.fit(data=data,
                  ts_length=ts_length,
                  r=r,
                  mode_selection=mode_selection,
                  lambda_min=lambda_min,
                  lambda_max=lambda_max)
    
    # Retrieve the learned DMD/DMDc model parameters if necessary
    if hasattr(model, 'get_dmd'):
        model.get_dmd()
    
    # Initialize prediction list
    prediction_list = []
    
    # Touring DMD/DMDc for every timestep
    # Start from ts_length to ensure we have enough data for the initial window
    for t in range(ts_length, len(dataset) - 1):
        # Prepare future control input
        control_future = dataset.get_control_inputs(t + 1).reshape(-1, 1)  # Shape: (m*m, 1)
        
        # Predict the next state
        if model.category() == 'DMDc':
            prediction = model.predict_future(t, control_future)
        prediction = model.predict_future(t)
        
        # Append prediction to the list
        prediction_list.append(prediction)
        print(f"{t}th timestep prediction finished.")
    
    # Shape: (num_samples - ts_length - 1, m*m)
    return data[ts_length + 1:], np.array(prediction_list) # Original data, Predicted Data


def MAELoss(y, y_hat):
    """
    Calculating Mean Absolute Error.

    Args:
        y (numpy.ndarray): Exact data for validation.
        y_hat (numpy.ndarray): Predicted data by DMD algorithm.

    Raises:
        ValueError: Occurs when shape of inputs are different.

    Returns:
        float: MAE Loss between y and y_hat.
    """
    if y.shape != y_hat.shape:
        raise ValueError("Inputs must have the same shape.")
    return np.mean(np.abs(y - y_hat))


def linkAccuracy(y, y_hat):
    """
    Custom loss function to measure link accuracy:
    linkAccuracy = (total_link_num - wrong_link_num) / total_link_num

    Args:
        y (numpy.ndarray): Exact adjacency matrix for validation.
        y_hat (numpy.ndarray): Predicted adjacency matrix by DMD algorithm.

    Raises:
        ValueError: Occurs when shape of inputs are different.

    Returns:
        float: Calculated linkAccuracy between y and y_hat.
    """
    if y.shape != y_hat.shape:
        raise ValueError("Inputs must have the same shape.")

    total_link_num = y.size
    # Identify mismatches
    wrong_links = np.sum((y != 0) != (y_hat != 0))
    
    return (total_link_num - wrong_links) / total_link_num


def disconnector(adjs, threshold, binary=False):
    """
    If value in adjacency matrix is too small, it is considered as disconnected.

    Args:
        adjs (numpy.ndarray): Adjacency matrices to process.
        threshold (float): A threshold to determine whether to connect.
        binary (bool, optional): Whether to binarize the adjacency matrix. Default is False.

    Returns:
        numpy.ndarray: Processed Adjacency matrices.
    """
    adjs_processed = adjs.copy()
    adjs_processed[adjs_processed < threshold] = 0
    if binary:
        adjs_processed[adjs_processed >= threshold] = 1

    return adjs_processed


def lossPlot(loss_list, scatter=False, title="Loss Plot", xlabel="Timestep", ylabel="Loss"):
    """
    Plotting losses for every timestep.

    Args:
        loss_list (list or numpy.ndarray): Loss list of every timestep.
        scatter (bool, optional): Whether to plot scatter. Default is False.
        title (str, optional): Title of the plot. Default is "Loss Plot".
        xlabel (str, optional): X-axis label. Default is "Timestep".
        ylabel (str, optional): Y-axis label. Default is "Loss".
    """ 
    plt.figure(figsize=(16, 9))
    if scatter:
        plt.scatter(range(len(loss_list)), loss_list, color='blue', alpha=0.6)
    else:
        plt.plot(range(len(loss_list)), loss_list, color='blue', linewidth=2)
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.grid(True)
    plt.show()


def link_optimizer(y, y_hat, binary=False):
    """
    Optimizing threshold for maximizing LinkAccuracy.

    Args:
        y (numpy.ndarray): Exact dataset with time steps.
        y_hat (numpy.ndarray): Predicted dataset with time steps.
        binary (bool, optional): Whether dataset is binary or weighted. Default is False.

    Returns:
        float: Optimal Threshold that maximizes linkAccuracy
    """
    edge = round(np.sqrt(y.shape[1]))
    thresholds = np.linspace(y_hat.min(), y_hat.max(), num=200)

    accuracyList = []
    for threshold in thresholds:
        _y_hat = disconnector(y_hat.copy(), threshold, binary)
        # Reshape all adjacency matrices
        _y_hat_reshaped = _y_hat.reshape(_y_hat.shape[0], edge, edge)
        y_reshaped = y.reshape(y.shape[0], edge, edge)
        # Compute linkAccuracy for all samples
        accuracies = np.array([linkAccuracy(y_reshaped[i], _y_hat_reshaped[i]) for i in range(y.shape[0])])
        accuracyList.append(accuracies.mean())

    # Get the threshold that maximizes the average accuracy
    return thresholds[np.argmax(accuracyList)] # Optimal Threshold

def get_loss_list(DMD_Class, dataset, ts_length,
                  r=None,
                  mode_selection=False,
                  lambda_min=0.9,
                  lambda_max=1.1,
                  binary=False):
    """
    Get list of MAE, Link loss for every timestep.

    Args:
        DMD_Class (class): Class implementation of DMD or DMDc algorithm.
        dataset (class): Graph Dataset class.
        path (str): Path to the dataset.
        ts_length (int): Length of time series window.
        r (int, optional): Rank for SVD truncation.
        mode_selection (bool, optional): Whether to perform mode selection. Default is False.
        lambda_min (float, optional): Minimum threshold for eigenvalue magnitude. Used if mode_selection is True.
        lambda_max (float, optional): Maximum threshold for eigenvalue magnitude. Used if mode_selection is True.
        binary (bool, optional): Whether dataset is binary or weighted. Default is False.

    Returns:
        tuple: (MAE_list, link_list, optimal_threshold)
    """
    exact, predict = DMD_touring(DMD_Class, dataset, ts_length,
                                 r=r,
                                 mode_selection=mode_selection,
                                 lambda_min=lambda_min,
                                 lambda_max=lambda_max)
    edge = round(np.sqrt(len(exact[0])))

    threshold = link_optimizer(exact, predict, binary)

    MAE_list = []
    link_list = []
    for i in range(len(predict)):
        disconnected = disconnector(predict[i].copy(), threshold, binary)
        MAE_list.append(MAELoss(exact[i], disconnected))
        link_list.append(linkAccuracy(exact[i].reshape(edge, edge), disconnected.reshape(edge, edge)))

    return MAE_list, link_list, threshold


def graphViz(adj_matrix, title="Graph Visualization", node_labels=None, layout='circular'):
    """
    Visualize Graph data.

    Args:
        adj_matrix (list or numpy.ndarray): An adjacency matrix of graph to visualize.
        title (str, optional): Title of the graph. Default is "Graph Visualization".
        node_labels (list, optional): List of node labels. Default is None.
        layout (str, optional): Layout type ('circular', 'spring', 'shell', etc.). Default is 'circular'.
    """
    G = nx.Graph()
    G = nx.from_numpy_array(adj_matrix)
    
    if layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'spring':
        pos = nx.spring_layout(G)
    elif layout == 'shell':
        pos = nx.shell_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.circular_layout(G)  # Default

    plt.figure(figsize=(10, 10))

    nx.draw(
        G, pos, with_labels=True, labels=node_labels, node_color='lightblue', edge_color='gray', 
        node_size=1000, font_size=10, font_color='black'
    )

    # Show weight label of edge
    edge_labels = nx.get_edge_attributes(G, "weight")
    # Set format of weight label 
    edge_labels = {edge: f"{weight:.3f}" for edge, weight in edge_labels.items()} 
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Visualize Graph
    plt.title(title, fontsize=20)
    plt.show()