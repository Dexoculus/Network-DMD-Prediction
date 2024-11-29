from Module.DMD import DMD
from Module.face2face import Dataset
from Module.utils import *

# Path of dataset files
w_path = './data/resistance/network0_weighted.csv'
path = './data/resistance/network0.csv'

# Initialize Class
dmd = DMD
dataset = Dataset

exact, predict = DMD_touring(dmd, dataset, w_path, ts_length=1, r=10)

MAE, Link = get_loss_list(dmd, dataset, w_path, ts_length=1, r=10)

lossPlot(MAE)
lossPlot(Link)

graphViz(predict[7000].reshape(7, 7))
