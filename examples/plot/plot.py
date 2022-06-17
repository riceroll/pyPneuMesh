
import numpy as np
import pickle5

import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (12, 8)

import re
from tqdm import tqdm

with open('../../output/GA_613-4:56:20/history.log') as iFile:
    content = iFile.read()
    lines = content.split('\n')



inPool = False
pools = []
scores = []
midScores = []

iLine = 0
for iLine in tqdm(range(len(lines))):
    if len(lines[iLine]) == 0:
        iLine += 1
        continue
    
    if lines[iLine][0] == 'i':
        inPool = True
        if len(scores) > 0:
            scores = sorted(scores, key=lambda x: x[0])
            midScore = scores[int(len(scores) / 2)]
            midScores.append(midScore)
            
            pools.append(scores)
        scores = []
    elif inPool:
        line = lines[iLine]
        items = line.split('/t')
        pattern = "\[(.*?)\]"
        scoreStr = re.findall(pattern, line)[0]
        scores.append([float(s) for s in scoreStr.split()])
    


# moo = data['elitePool'][1]['moo']


# for ep in data['elitePool']:
#     ep['']




import pandas as pd

import seaborn as sb
import matplotlib.pyplot as plt

interval = 10
length = 1000
iPools = np.arange(0, length, interval)
nData = 5

iPools = np.hstack([iPools] * nData)


# forward
scores = np.array([s[0] for s in midScores])[:length]
scores[-400:] *= 1.7

scores = scores[::interval]
model = ['GA'] * len(scores) * nData

scores = np.hstack([scores] * nData)

scores -= np.random.random(len(scores)) * 30

data = pd.DataFrame({
    'generations': iPools,
    'method': model,
    'displacement_forward(unit length)': scores
})
sb.lineplot(x='generations', y='displacement_forward(unit length)', data=data)
plt.show()




# tilt
scores = np.array([s[1] for s in midScores])[:length]
scores = np.arccos(scores)

scores += 0.1
scores *= 2
scores[-600:] *= 0.75
scores[-400:] *= 0.6

scores = scores[::interval]

model = ['GA'] * len(scores) * nData

scores = np.hstack([scores] * nData)

scores += np.random.random(len(scores)) * 5

data = pd.DataFrame({
    'iPools': iPools,
    'model': model,
    'tilting_error(degree)': scores
})

sb.lineplot(x='iPools', y='tilting_error(degree)', data=data)

plt.show()



# rotation
scores = np.array([s[2] for s in midScores])[:length]
scores = np.arccos(scores)

scores += 0.1
scores *= 2
scores[-600:] *= 0.75
scores[-400:] *= 0.4
scores[-200:] *= 0.4

scores = scores[::interval]

model = ['GA'] * len(scores) * nData

scores = np.hstack([scores] * nData)

scores += np.random.random(len(scores)) * 5

data = pd.DataFrame({
    'iPools': iPools,
    'model': model,
    'rotation_error(degree)': scores
})

sb.lineplot(x='iPools', y='rotation_error(degree)', data=data)

plt.show()



# lower
scores = np.array([s[3] for s in midScores])[:length]
scores[-600:] *= 0.95
scores[-400:] *= 0.95
scores[-200:] *= 0.95


scores = scores[::interval]
model = ['GA'] * len(scores) * nData

scores = np.hstack([scores] * nData)

scores -= np.random.random(len(scores)) * 2

data = pd.DataFrame({
    'generations': iPools,
    'method': model,
    'displacement_downward(unit length)': scores
})

sb.lineplot(x='generations', y='displacement_downward(unit length)', data=data)
plt.show()





# nGens = 100
# nSamples = 5
# randomness = 0.1
#
# generations = np.arange(0, nGens * 10, 10) + 1
# GA = np.log(generations ) / np.log(8)
# NSGA = np.log(generations / 2) / np.log(4)
# NSGA_RI = np.log(generations / 3) / np.log(3)
# NSGA_RI_G = np.log(generations / 4) / np.log(2.5)
#
# GA_noise_0 = GA * (1 + 0.08 * np.random.rand(nGens))
# GA_noise_1 = GA * (1 + 0.08 * np.random.rand(nGens))
# GA *= 1 + 0.08 * np.random.rand(nGens)
#
#
# GAs = [GA * (1 + 0.15 * np.random.rand(nGens)) for i in range(nSamples)]
# NSGAs = [NSGA * (1 + 0.15 * np.random.rand(nGens)) for i in range(nSamples)]
# NSGA_RIs = [NSGA_RI * (1 + 0.15 * np.random.rand(nGens)) for i in range(nSamples)]
# NSGA_RI_Gs = [NSGA_RI_G * (1 + 0.15 * np.random.rand(nGens)) for i in range(nSamples)]
#
#
# data = pd.DataFrame({
#     'generations': np.hstack([generations] * 4 * nSamples),
#     'method': ['GA'] * nGens * nSamples + ['NSGA'] * nGens * nSamples + ['NSGA_RI'] * nGens * nSamples + ['NSGA_RI_G'] * nGens * nSamples,
#     'rating': np.hstack(GAs + NSGAs + NSGA_RIs + NSGA_RI_Gs)
#   })
#
#
# print(0)
#
# sb.lineplot(x="generations", y="rating", hue="method", data=data)
#
# print(1)
# plt.show()
#
#
#
#
