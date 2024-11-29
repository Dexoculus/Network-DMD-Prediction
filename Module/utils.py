import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# from Module.DMD import DMD
# from Module.face2face import Dataset

def DMD_touring(DMD, Dataset, path, ts_length, r=None):
    """
    Args:
        DMD (class) : Class implementation of DMD algorithm.
        Dataset (class) : Graph Dataset.
        ts_length (int) : length of time series window.
        r (int) : Number of singular value to compute.

    Returns:
        Original data and predicted data
    """
    # Initialize Classes
    _data = Dataset(path)
    data = np.array([_data[i].flatten(order='F') for i in range(len(_data))])
    dmd = DMD()

    # Fitting DMD Model
    dmd.fit(data=data, ts_length=ts_length, r=r)
    prediction_list = []

# Touring DMD for every timestep
    prediction_list = []
    for t in range(ts_length+1, len(_data)):
        prediction_list.append(dmd.predict_future(t))

    return np.array(data[ts_length+1:]), np.array(prediction_list)

def MAELoss(y, y_hat):
    """
    Calculating Mean Absolute Error.

    Args:
        y (numpy.ndarray) : Exact data for validation.
        y_hat (numpy.ndarray) : Predicted data by DMD algorithm.

    Raises:
        ValueError
            Accurs when shape of inputs are different.

    Returns:
        MAE Loss between y and y_hat.
    """
    if y.shape != y_hat.shape:
        raise ValueError("Inputs must have same shape.")
    return np.abs(sum(y - y_hat)) / len(y)

def disconnector(adj, threshold):
    """
    If value in adjacency matrix is too small, It is considered as disconnected.

    Args:
        adj (numpy.ndarray) : An adjacency matrix to process.
        threshold (float) : A Threshold to determine whether to connect.
    """
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i][j] < threshold:
                adj[i][j] = 0

    return adj

def linkLoss(y, y_hat):
    """
    Novel defined loss for validate link between edges in two data:
        linkLoss = (total_link_num - wrong_link_num) / total_link_num

    Args:
        y (numpy.ndarray) : Exact adjacency matrix for validation.
        y_hat (numpy.ndarray) : Predicted adjacency matrix by DMD algorithm.

    Raises:
        ValueError
            Accurs when shape of inputs are different.

    Returns:
        Calculated linkLoss beteen y and y_hat.
    """
    if y.shape != y_hat.shape:
        raise ValueError("Inputs must have same shape.")
    
    total_link_num = y.shape[0] * y.shape[1]
    wrong_link_num = 0
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if y[i][j]!=0 and y_hat[i][j]!=0:
                pass
            elif y[i][j]==0 and y_hat[i][j]==0:
                pass
            else:
                wrong_link_num += 1

    return (total_link_num-wrong_link_num) / total_link_num

def get_loss_list(DMD, Dataset, path, ts_length, r=None):
    """
    Get list of MAE, Link loss for every timestep.

    Args:
        DMD (class) : Class implementation of DMD algorithm.
        Dataset (class) : Graph Dataset.
        ts_length (int) : length of time series window.
        r (int) : Number of singular value to compute.

    Returns:
        Loss list of MAE and link loss.
    """
    exact, predict = DMD_touring(DMD, Dataset, path, ts_length, r)

    MAE_list = []
    link_list = []
    for i in range(len(predict)):
        MAE_list.append(MAELoss(exact[i], predict[i]))
        link_list.append(linkLoss(exact[i].reshape(7, 7), predict[i].reshape(7, 7)))

    return MAE_list, link_list

def lossPlot(loss_list):
    """
    Plotting losses for every time steps.

    Args:
        lost_list (list or numpy.ndarray) : loss list of every timestep.
    """ 
    plt.figure(figsize=(16, 9))
    plt.plot(loss_list)
    plt.xlabel("timestep (s)")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

def graphViz(adj_matrix):
    """
    Visualize Graph data.

    Args:
        adj_matrix (list or numpy.ndarray) : adjacency matrix of graph to visualize.
    """
    G = nx.Graph()
    G = nx.from_numpy_array(adj_matrix)
    pos = nx.circular_layout(G)
    plt.figure(figsize=(10, 10))

    nx.draw(
    G, pos, with_labels=True, node_color='lightblue', edge_color='gray', 
    node_size=1000, font_size=10, font_color='black'
    )

    # Show weight label of edge
    edge_labels = nx.get_edge_attributes(G, "weight")
    # Set format of weight label 
    edge_labels = {edge: f"{weight:.3f}" for edge, weight in edge_labels.items()} 
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Visualize Graph
    plt.title("Graph Visualization with Weights")
    plt.show()