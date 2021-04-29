

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_heat(x):
    # f, ax1 = plt.subplots(figsize=(4, 4), nrows=1)
    x=(x-np.min(x))/(np.max(x)-np.min(x))
    # x=(x-np.mean(x))/(np.std(x))
    f, ax1 = plt.subplots()
    # cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
    # sns.heatmap(x,  ax=ax1,cmap = cmap)
    sns.heatmap(x,  ax=ax1,cmap = 'rainbow')
    plt.axis('off')
    plt.show()
    # plt.waitforbuttonpress()
import os
def save_heat(x,dir,name):
    # f, ax1 = plt.subplots(figsize=(4, 4), nrows=1)
    x=(x-np.min(x))/(np.max(x)-np.min(x))
    f, ax1 = plt.subplots()
    sns.heatmap(x,  ax=ax1,cmap='rainbow')
    plt.axis('off')
    plt.savefig(os.path.join(dir,name),dpi=400)

if __name__=="__main__":
    plot_heat()

