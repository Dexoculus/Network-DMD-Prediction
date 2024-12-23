from GraphDMD.DMD import DMD
from GraphDMD.DMD import Dataset
from GraphDMD.utils import *

def main():
    w_path = './data/resistance/network0_weighted.csv'
    path = './data/resistance/network0.csv'
    
    dmd = DMD
    dataset = Dataset(path)
    
    ts_length = 1
    r = 30
    
    exact, predict = DMD_touring(dmd, dataset, ts_length, r, mode_selection=True)
    MAE, link, threshold = get_loss_list(dmd, dataset, ts_length, r, mode_selection=True, binary=False)

    predict_means = []
    for i in predict:
        predict_means.append(np.mean(i))
    
    print(f"Optimal Threshold: {threshold}")
    print(f"Average MAE: {np.mean(MAE)}")
    print(f"Average Link Accuracy: {np.mean(link)}")
    
    edge = round(np.sqrt(exact.shape[1]))
    
    lossPlot(predict_means, scatter=False, title="Average Value of every Prediction", xlabel="Timestep", ylabel="Mean")
    # Plot Loss
    lossPlot(MAE, scatter=False, title="MAE Loss over Time", xlabel="Timestep", ylabel="MAE")
    lossPlot(link, scatter=True, title="Link Accuracy over Time", xlabel="Timestep", ylabel="Link Accuracy")
    
    graphViz(predict[3000].reshape((edge, edge)), title="DMDc Predicted Adjacency Matrix at First Prediction")
    

if __name__ == "__main__":
    main()