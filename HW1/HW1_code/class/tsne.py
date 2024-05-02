import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
import random
import tqdm
import utils
import train_q2
import __main__

setattr(__main__,"ResNet",train_q2.ResNet)

CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                   'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                   'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

device="cuda"

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

model = torch.load('/home/ubuntu/hw1/checkpoints/checkpoint-last-2.pth').to(device)
    
# model = torch.nn.Sequential(*list(model.children())[:-1])

model.eval()

test_loader = utils.get_data_loader('voc', train=False, batch_size=1000, split='test', inp_size=224)

# labels = []
feats = None
color = []

with torch.no_grad():
    for batch_idx, (data, target, wgt) in enumerate(test_loader):
        # data, target, wgt = data, target, wgt
        data , target, wgt = data.to(device),target.to(device),wgt.to(device)

        curr_feats = model.resnet(data).cpu().numpy()

        if feats == None:
            feats = curr_feats
        else:
            feats = np.concatenate((feats,curr_feats))

        lab = target*wgt

        label = lab.cpu().numpy()

        for l in label:
            idx = np.mean(np.where(l==1)[0])
            color.append(idx)

        break

        # for l in label:

# print(label.shape)
tsne = TSNE(n_components=2)
tsne_result = tsne.fit_transform(feats)

def scale_to_plot(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range

tx = scale_to_plot(tsne_result[:, 0])
ty = scale_to_plot(tsne_result[:, 1])

legend_lab = list(CLASS_NAMES)

legend_values = list(np.linspace(0,len(CLASS_NAMES)-1, len(CLASS_NAMES),dtype=int))
# viridis = matplotlib.colormaps.get_cmap('viridis', 20)
viridis = cm.get_cmap('viridis', 20)

classes_plt = []
for i in range(0,len(legend_lab)):
    class_idx = matplotlib.patches.Patch(color=viridis(legend_values[i]),label=legend_lab[i])
    classes_plt.append(class_idx)

# plt.scatter(tx,ty)

# print(classes_plt)

fig, axs = plt.subplots(figsize = (15,15))

axs.scatter(tx, ty, c=color, cmap='viridis')
axs.legend(handles=classes_plt, loc = 'best')
plt.show()
plt.savefig('/home/ubuntu/hw1/q1_q2_classification/tsne_v3.png', format = 'png')




