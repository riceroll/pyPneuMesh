
import numpy as np
import pickle5

data = pickle5.load(open('./output/GA_531-8-36-53/iPool_580','rb'))
moo = data['elitePool'][1]['moo']


# for ep in data['elitePool']:
#     ep['']



#
# import pandas as pd
#
# import seaborn as sb
# import matplotlib.pyplot as plt
#
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
