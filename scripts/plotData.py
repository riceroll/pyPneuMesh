import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from pyPneuMesh.Model import Model

# sns.set_theme(style="darkgrid")
sns.set_theme(style="white")

# plt.gca().set_facecolor((0.95,0.95,0.95, 1.0))


matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Helvetica Neue'
matplotlib.rcParams['font.size'] = 7


# plt.figure(figsize=)
# fig = plt.figure(figsize=(2, 2))  # Create a new figure
# print( fig.get_dpi() ) # Get the DPI for the figure)

#
# def plotCurve(x, y, variances=None, title='hehe', x_axis='iteration', y_axis='performance', x_interval=25, n_yticks=10,
#               color='b', label='fitness'):
#     plt.plot(x, y, label=label, color=color, linewidth=2.0)  # Add a label
#
#     # Add the shaded variance
#     if variances is not None:
#         upper = y + variances
#         lower = y - variances
#         plt.fill_between(x, lower, upper, color=color, alpha=.1)
#
#     plt.tick_params(left="on", bottom="on", length=4, width=1.0)
#
#     plt.xticks(np.arange(min(x), max(x), x_interval))
#     y_locs = np.arange(min(y), max(y) + (max(y) - min(y)) * 0.2, (max(y) - min(y)) / n_yticks)
#     plt.yticks(y_locs, [f'{loc:.3f}' for loc in y_locs])
#
#     # Change the frame width
#     ax = plt.gca()  # Get the current axes
#     for axis in ['top', 'bottom', 'left', 'right']:
#         ax.spines[axis].set_linewidth(1.0)  # Set the line width to 2.0
#
#     plt.title(title)
#     plt.xlabel(x_axis)
#     plt.ylabel(y_axis)
#
#     plt.legend(frameon=False)
#
#
# folder = '/Users/Roll/Desktop/dataPlotting/'
#
# colors = [
#     '#F21365',
#     '#66C8D2',
#     '#F1B807',
#     '#F5AB55',
#     '#F27A5E',
#     '#63A5BF'
# ]
#
# table_8_0 = 'table_8_0.npy'
# table_8_1 = 'table_8_1.npy'
# table_8_2 = 'table_8_2.npy'
#
#
# # name0 = 'lobster.npy'
#
#
# def plot(name0, index, title, label, x_axis, y_axis, color, name1=None, name2=None):
#     data = np.load(folder + name0, allow_pickle=True).all()
#     x0 = data['scores'].keys()
#     y0 = [data['scores'][xx][index] for xx in x0]
#     #
#     data = np.load(folder + name1, allow_pickle=True).all()
#     x1 = data['scores'].keys()
#     y1 = [data['scores'][xx][index] for xx in x1]
#
#     data = np.load(folder + name2, allow_pickle=True).all()
#     x2 = data['scores'].keys()
#     y2 = [data['scores'][xx][index] for xx in x2]
#
#     y00 = []
#     y11 = []
#     y22 = []
#     yy0 = y0[0]
#     yy1 = y1[0]
#     yy2 = y2[0]
#
#     i0 = 0
#     i1 = 0
#     i2 = 0
#     for i in range(max(x0) + 1):
#         if i in x0:
#             yy0 = y0[i0]
#             i0 += 1
#
#         y00.append(yy0)
#
#         if i in x1:
#             yy1 = y1[i1]
#             i1 += 1
#
#         y11.append(yy1)
#
#         if i in x2:
#             yy2 = y2[i2]
#             i2 += 1
#
#         y22.append(yy2)
#
#     x = np.arange(max(x0) + 1)
#     y0 = np.array(y00)
#     y1 = np.array(y11)
#     y2 = np.array(y22)
#
#     y = np.array([y0, y1, y2]).mean(0)
#     variance = np.array([y0, y1, y2]).std(0) ** 2
#
#     print(data['objectives'])
#
#     plotCurve(x, y, variance, title=title,
#               x_axis=x_axis, y_axis=y_axis, x_interval=100, n_yticks=10, color=color, label=label)
#
#
# plot(table_8_0, 1, title='4-legged robot locomotion', x_axis='iterations', y_axis='displacement(m)', color=colors[0],
#      label='hehe', name1=table_8_1, name2=table_8_2)
# plot(table_8_0, 3, title='4-legged robot locomotion', x_axis='iterations', y_axis='displacement(m)', color=colors[1],
#      label='hehe', name1=table_8_1, name2=table_8_2)
#
# plt.show()
#
# # plotCurve(x, y0, None, title='Lobster Energy Efficiency Training Performance',
# #           x_axis='iterations', y_axis='Efficiency(m/kJ)', x_interval=100, y_interval=0.5, color='#F21365')
#






def plotData(model:Model):
    
    w = 12
    h = 3
    
    plt.figure(figsize=(w,h))
    # fig = plt.figure(figsize=(2, 2))  # Create a new figure
    # print( fig.get_dpi() ) # Get the DPI for the figure)
    
    
    
    colors = [
        '#F21365',
        '#66C8D2',
        '#F1B807',
        '#F5AB55',
        '#F27A5E',
        '#63A5BF'
    ]
    
    indices = np.arange(len(model.e))
    
    values = model.contractionLevel
    
    plt.tick_params(left="on", bottom="on", length=4, width=1.0)
    
    plt.xticks(np.arange(min(indices), max(indices), 20))
    y_locs = np.arange(min(values), max(values)+1, 1)
    plt.yticks(y_locs, [f'{loc * 0.12:.1f}({i})' for i, loc in enumerate(y_locs)])
    
    # Change the frame width
    ax = plt.gca()  # Get the current axes
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.0)  # Set the line width to 2.0
    
    # plt.title('contraction')
    plt.xlabel('Edge index')
    plt.ylabel('Contraction ratio \n (Contraction index)')
    
    plt.legend(frameon=False)
    
    plt.bar(indices, values, width=0.8)

    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.45, top=0.85)
    plt.savefig('/Users/Roll/Desktop/MetaTruss-Submission/Figures/' + 'contractionDesign.png', dpi=1000)
    
    plt.show()
    
    
    
    plt.figure(figsize=(w,h))
    
    values = model.edgeChannel
    
    cs = [colors[value] for value in values]

    plt.xlabel('Edge index')
    plt.ylabel('Channel Index')
    

    y_locs = np.arange(min(values), max(values)+1, 1)
    plt.yticks(y_locs, [f'{loc}' for i, loc in enumerate(y_locs)])

    ax = plt.gca()  # Get the current axes
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.0)  # Set the line width to 2.0
    
    plt.bar(indices, 1.0, bottom=values, width=1.0, color=cs)

    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.45, top=0.85)
    
    plt.savefig('/Users/Roll/Desktop/MetaTruss-Submission/Figures/' + 'channelDesign.png', dpi=1000)
    
    plt.show()





import numpy as np
import json
import multiprocessing

from pyPneuMesh.utils import readNpy, readMooDict
from pyPneuMesh.Model import Model
from pyPneuMesh.Graph import Graph
from pyPneuMesh.MOO import MOO
from pyPneuMesh.GA import GA

mode = "start"
# mode = "continue"
# mode = "load"
mode = "configMOO"

GACheckpointDir = "/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/scripts/trainTable_2022-12-09_5-51/output/gcp_2022-12-09_12-19-30/ElitePool_390.gacheckpoint.npy"

if mode == "start":
    GASetting = {
        'nGenesPerPool': 512,
        'nSurvivedMin': 128,  # actually is max
        'nGensPerPool': 6,
        
        'nWorkers': multiprocessing.cpu_count(),
        
        'folderDir': 'scripts/trainTable_channel_6/',
        
        'contractionMutationChance': 0.01,
        'actionMutationChance': 0.01,
        'graphMutationChance': 0.1,
        'contractionCrossChance': 0.02,
        'actionCrossChance': 0.02,
        'crossChance': 0.5,
        
        'randomInit': False
    }
    ga = GA(GASetting=GASetting)
    ga.run()

elif mode == "continue":
    ga = GA(GACheckpointDir=GACheckpointDir)
    ga.run()

elif mode == "load":
    ga = GA(GACheckpointDir=GACheckpointDir)
    print('genePool')
    ga.logPool(ga.genePool, printing=True, showAllGenes=True, showRValue=True)
    print('elitePool')
    ga.logPool(ga.elitePool, printing=True, showAllGenes=True, showRValue=True)
    
    genes = []
    for gene in ga.elitePool:
        if 2 < gene['score'][0] < 4 and gene['score'][1] > -200 and \
                gene['score'][2] > 0.5 and \
                gene['score'][3] > -1.0 and gene['score'][4] > -0.01:
            # gene['score'][5] > 0.7:
            genes.append(gene)
    print(genes)


elif mode == "configMOO":
    mooDict = readMooDict('scripts/trainTable_2022-12-09_5-51/data/')
    moo = MOO(mooDict=mooDict)



plotData(moo.model)

    
