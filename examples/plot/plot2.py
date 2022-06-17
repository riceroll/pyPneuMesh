
import numpy as np
import pickle5

import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (12, 8)

import re
from tqdm import tqdm

params = [
    29 / 7,
    1,
    1
]

i_param = 1
ratio = params[i_param]

with open('./outputlog_'+str(i_param)) as iFile:
    content = iFile.read()
    lines = content.split('\n')

updates = []
means = []
maxes = []
types = []
values = []

for i in range(int(len(lines) / 2)):
    line_0 = lines[i * 2]
    line_1 = lines[i * 2 + 1]
    
    iUpdate = int(line_0.split()[1][:-1])
    items = line_1.split()
    
    meanmedium = items[6]
    minmax = items[9]
    mean, medium = meanmedium[:-1].split('/')
    mean = float(mean)
    medium = float(medium)
    minn, maxx = minmax.split('/')
    minn = float(minn)
    maxx = float(maxx)

    updates.append(iUpdate)
    if i_param == 1:
        mean = np.arccos(mean)
        medium = np.arccos(medium)
        maxx = np.arccos(maxx)
        minn = np.arccos(minn)
    
    values.append(mean * ratio)
    types.append('mean')

    updates.append(iUpdate)
    values.append(maxx * ratio)
    types.append('max')



import pandas as pd

import seaborn as sb
import matplotlib.pyplot as plt


data = pd.DataFrame({
    'updates': updates,
    'values': values,
    'types': types
})
sb.lineplot(x='updates', y='values', hue='types', data=data)
plt.show()

